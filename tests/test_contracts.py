from dataclasses import replace
from itertools import permutations, combinations
from contextlib import redirect_stdout
from unittest.mock import patch
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sepa_plan_b.config import Config, Geometry, Model, Data, Training, Probe
from sepa_plan_b.data import prepare, manifests, ImageTiles, epoch_batches
from sepa_plan_b.engine import train, load_checkpoint, load_encoder_run
from sepa_plan_b.evaluation import fit_linear, knn_accuracy, label_subset, probe, padding_diagnostic, spatial_probe, analyze
from sepa_plan_b.geometry import sample_layout, layout_from_mapping, split_tiles
from sepa_plan_b.losses import latent_loss, pair_targets, relation_loss
from sepa_plan_b.model import SEPA, TileEncoder
from sepa_plan_b.diagnostics import position_sensitivity
from sepa_plan_b.cli import main


def tiny(**kwargs):
    return Config(model=Model(name="tiny", predictor_dim=32, predictor_depth=1, predictor_heads=4),
                  training=Training(steps=4, warmup_steps=1, batch_size=2, checkpoint_every=1,
                                    log_every=4, device="cpu", num_workers=0),
                  probe=Probe(epochs=2, batch_size=8, spatial_steps=2), **kwargs)


class GeometryTests(unittest.TestCase):
    def test_crop_and_padding_preserve_all_pixels(self):
        image = torch.arange(224*224).reshape(1,224,224).repeat(3,1,1).float()
        for mode in ("reflect", "replicate", "constant"):
            tiles = split_tiles(image, Geometry("pad74", mode), normalize=False)
            recovered = tiles[..., 3:77, 3:77].reshape(3,3,3,74,74).permute(2,0,3,1,4).reshape(3,222,222)
            self.assertTrue(torch.equal(recovered, image[:,1:223,1:223]))
            self.assertEqual(F.unfold(tiles, 16, stride=16).shape[-1], 25)

    def test_padding_cannot_read_neighbor(self):
        image = torch.randn(3,224,224, requires_grad=True)
        tile = split_tiles(image, Geometry("pad74"), normalize=False)[0]
        tile.sum().backward()
        mask = image.grad.abs().sum(0) > 0
        expected = torch.zeros_like(mask)
        expected[1:75,1:75] = True
        self.assertTrue(torch.equal(mask, expected))

    def test_three_cycle_direction_and_anchor(self):
        p = [1,3,2,0,4,5,6,7,8]
        layout = layout_from_mapping([2,6], p)
        expected = [3,0,1,4,5,7,8]
        self.assertEqual(layout.canonical_ids.tolist(), expected)
        self.assertEqual(layout.moved_count, 3)

    def test_all_mask_sets_and_subset_permutations_are_valid(self):
        ring = [0,1,2,3,5,6,7,8]
        for masked in combinations(ring, 2):
            available = [x for x in ring if x not in masked]
            for assigned in permutations(available):
                p = list(range(9))
                for a,b in zip(available,assigned): p[a]=b
                layout = layout_from_mapping(masked,p)
                self.assertEqual(sorted(layout.canonical_ids.tolist()), sorted(set(range(9))-set(masked)))

    def test_k_pairing_and_distribution(self):
        moves = {k: [] for k in (0,2,3,4,5,6)}
        for sample in range(1200):
            arms = {k: sample_layout(7,2,str(sample),k) for k in moves}
            for k,l in arms.items():
                self.assertTrue(torch.equal(l.query_slots, arms[0].query_slots))
                self.assertEqual(l.piece_to_slot[4].item(),4)
                self.assertNotIn(4,l.query_slots.tolist())
                self.assertLessEqual(l.moved_count,k)
                moves[k].append(l.moved_count)
        for k,counts in moves.items():
            self.assertAlmostEqual(np.mean(counts), max(0,k-1), delta=0.12)
        self.assertAlmostEqual(moves[3].count(0)/len(moves[3]), 1/6, delta=0.04)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.tiles = torch.randn(2,9,3,80,80)
        self.slots = torch.tensor([[0,1,3,4,5,7,8]]).expand(2,-1)
        self.queries = torch.tensor([[2,6]]).expand(2,-1)
        self.visible = self.tiles[:,self.slots[0]]

    def test_forward_shapes_and_hidden_pixels(self):
        for method in ("sepa","full"):
            model = SEPA(tiny(), method=method).eval()
            with torch.no_grad():
                predicted,h,_ = model(self.visible,self.slots,self.queries)
                target = model.targets(self.tiles,self.queries)
                changed = self.tiles.clone()
                changed[:,self.queries[0]] += torch.randn_like(changed[:,self.queries[0]])
                other = model.targets(changed,self.queries)
                again = model(changed[:,self.slots[0]],self.slots,self.queries)[0]
            self.assertEqual(tuple(predicted.shape),(2,2,32))
            self.assertEqual(tuple(h.shape),(2,7,32))
            self.assertTrue(torch.equal(predicted,again))
            self.assertFalse(torch.equal(target,other))

    def test_local_encoder_no_cross_tile_gradients(self):
        model = SEPA(tiny())
        pixels = self.visible.clone().requires_grad_()
        model.encoder(pixels)[0,0,0].backward()
        self.assertGreater(pixels.grad[0,0].abs().sum(),0)
        self.assertEqual(pixels.grad[0,1:].abs().sum().item(),0)
        self.assertEqual(pixels.grad[1].abs().sum().item(),0)

    def test_position_binding_and_order(self):
        model = SEPA(tiny()).eval()
        order = torch.tensor([2,0,1,3,4,5,6])
        with torch.no_grad():
            a = model(self.visible,self.slots,self.queries)[0]
            b = model(self.visible[:,order],self.slots[:,order],self.queries)[0]
            c = model(self.visible[:,order],self.slots,self.queries)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-5,rtol=1e-5))
        self.assertGreater((a-c).abs().max().item(),1e-6)

    def test_relation_head_does_not_change_shared_initialization(self):
        base = SEPA(tiny(),seed=9)
        for mode in ("undirected","directed","aggregation"):
            cfg = replace(tiny(),model=replace(tiny().model,relation=mode,relation_weight=0.5))
            changed = SEPA(cfg,seed=9)
            for name in ("encoder","predictor"):
                for a,b in zip(getattr(base,name).parameters(),getattr(changed,name).parameters()):
                    self.assertTrue(torch.equal(a,b))

    def test_relation_isolation_and_gradient(self):
        for mode in ("undirected","directed"):
            cfg = replace(tiny(),model=replace(tiny().model,relation=mode,relation_weight=0.5))
            model = SEPA(cfg).train()
            _, h, logits = model(self.visible,self.slots,self.queries)
            loss = relation_loss(logits,self.slots,mode=="directed")
            loss.backward()
            self.assertGreater(model.encoder.patch_embed.weight.grad.abs().sum(),0)
            self.assertTrue(all(p.grad is None for p in model.predictor.parameters()))
            self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
            before = logits.detach()
            with torch.no_grad():
                model.predictor.slot_pe.normal_()
            self.assertTrue(torch.equal(before,model(self.visible,self.slots,self.queries)[2]))

    def test_semantic_loss_teacher_and_ema(self):
        model = SEPA(tiny()).train()
        p,_,_ = model(self.visible,self.slots,self.queries)
        t = model.targets(self.tiles,self.queries)
        loss = latent_loss(p,t)
        loss.backward()
        self.assertTrue(all(x.grad is None for x in model.teacher.parameters()))
        self.assertFalse(model.teacher.training)
        self.assertGreater(model.encoder.patch_embed.weight.grad.abs().sum(),0)
        self.assertTrue(torch.allclose(loss,latent_loss(p*2,t),atol=1e-6))
        with torch.no_grad():
            before = next(model.teacher.parameters()).clone()
            next(model.encoder.parameters()).add_(1)
            target = before*0.9+next(model.encoder.parameters())*0.1
            model.update_teacher(0.9)
        self.assertTrue(torch.allclose(next(model.teacher.parameters()),target,atol=1e-7))

    def test_pair_labels(self):
        adjacent,direction,valid = pair_targets(torch.arange(9)[None])
        self.assertEqual(adjacent.sum().item(),24)
        self.assertEqual(valid.sum().item(),72)
        self.assertEqual(len(direction[valid].unique()),8)
        self.assertFalse(adjacent[0,0,2])
        self.assertNotEqual(direction[0,0,1],direction[0,1,0])

    def test_position_diagnostic_preserves_state(self):
        model = SEPA(tiny()).train()
        rng = torch.get_rng_state().clone()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        values = position_sensitivity(model, self.tiles)
        self.assertGreater(values["position_delta_rms"], 1e-6)
        self.assertTrue(model.training)
        self.assertFalse(model.teacher.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k,v in before.items()))

    def test_aggregation_control_has_explicit_gradient_boundary(self):
        cfg = replace(tiny(), model=replace(tiny().model, relation="aggregation", relation_weight=0.5))
        model = SEPA(cfg)
        predicted, _, relations = model(self.visible, self.slots, self.queries)
        latent_loss(predicted, model.targets(self.tiles, self.queries)).backward()
        self.assertGreater(model.aggregation[0].weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model.relation_head.parameters()))
        model.zero_grad(set_to_none=True)
        relations = model(self.visible, self.slots, self.queries)[2]
        relation_loss(relations, self.slots).backward()
        self.assertGreater(model.relation_head.net[0].weight.grad.abs().sum(), 0)
        self.assertGreater(model.encoder.patch_embed.weight.grad.abs().sum(), 0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        for split in ("train","val"):
            for c in ("a","b"):
                path = self.root/split/c
                path.mkdir(parents=True)
                for i in range(4):
                    Image.fromarray(rng.integers(0,256,(100,120,3),dtype=np.uint8)).save(path/f"{i}.png")
        prepare(self.root/"train",self.root/"val",self.root/"manifests",expected_classes=2)
        self.cfg = replace(tiny(),data=Data(str(self.root/"manifests/train.json"),str(self.root/"manifests/val.json"),2),
                           training=replace(tiny().training,output_root=str(self.root/"runs")))

    def tearDown(self):
        self.temp.cleanup()

    def test_dataset_pairing_across_k_and_worker_count(self):
        from torch.utils.data import DataLoader
        tr,_ = manifests(self.cfg)
        a,b = ImageTiles(tr,self.cfg,5,0,True),ImageTiles(tr,self.cfg,5,3,True)
        x,y = a[(2,7)],b[(2,7)]
        self.assertTrue(torch.equal(x["canonical"],y["canonical"]))
        self.assertTrue(torch.equal(x["query_slots"],y["query_slots"]))
        def read(workers):
            loader=DataLoader(b,batch_sampler=list(epoch_batches(len(b),2,5,0)),num_workers=workers)
            return [(z["id"],z["canonical"],z["canonical_ids"]) for z in loader]
        for x,y in zip(read(0),read(2)):
            self.assertEqual(x[0],y[0]); self.assertTrue(torch.equal(x[1],y[1])); self.assertTrue(torch.equal(x[2],y[2]))

    def test_modified_image_rejected(self):
        tr,_=manifests(self.cfg)
        image=Path(tr["root"])/tr["records"][0]["path"]
        image.write_bytes(image.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError,"Image changed"):
            ImageTiles(tr,self.cfg)[0]

    def test_training_logs_constant_encoder_collapse(self):
        original = TileEncoder.forward
        def constant(encoder, tiles):
            return original(encoder, tiles) * 0 + 1
        with patch.object(TileEncoder, "forward", constant), redirect_stdout(io.StringIO()):
            run = train(self.cfg, stop_after=1)
        row = json.loads((Path(run["run_dir"]) / "history.jsonl").read_text())
        self.assertTrue(row["collapse_flag"])
        self.assertIn("low_encoder_std", row["collapse_reasons"])
        self.assertIn("low_target_std", row["collapse_reasons"])

    def test_invalid_encoder_output_does_not_create_probe_artifacts(self):
        with redirect_stdout(io.StringIO()):
            run = train(self.cfg, stop_after=1)
        output = self.root / "invalid_probe.json"
        with patch.object(SEPA, "encode", return_value=torch.full((8, 9, 32), float("nan"))):
            with self.assertRaises(FloatingPointError):
                probe(run["checkpoint"], output=output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".pt").exists())

    def test_train_val_duplicate_rejected(self):
        (self.root/"val/a/0.png").write_bytes((self.root/"train/a/0.png").read_bytes())
        with self.assertRaisesRegex(ValueError,"identical image"):
            prepare(self.root/"train",self.root/"val",self.root/"other",expected_classes=2)

    def test_resume_matches_continuous_and_probes_run(self):
        first=train(self.cfg,seed=5,k=3,stop_after=2)
        interrupted=load_checkpoint(first["checkpoint"])
        self.assertEqual(interrupted["step"],2)
        resumed=train(self.cfg,seed=5,k=3,resume=True)
        other=replace(self.cfg,training=replace(self.cfg.training,output_root=str(self.root/"continuous")))
        continuous=train(other,seed=5,k=3)
        a,b=load_checkpoint(resumed["checkpoint"]),load_checkpoint(continuous["checkpoint"])
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key],b["model"][key]),key)
        with self.assertRaises(FileExistsError): train(self.cfg,seed=5,k=3)
        result=probe(resumed["checkpoint"])
        self.assertEqual(result["n_train_labels"],8)
        self.assertEqual(result["n_val"],8)
        summary=analyze([result["output"]])
        self.assertEqual(summary["status"],"INSUFFICIENT")
        spatial=spatial_probe(resumed["checkpoint"],limit=0)
        self.assertEqual(spatial["n_train_images"]+spatial["n_test_images"],8)
        self.assertIn("canonical_position", spatial["results"]["prior"])

    def test_padding_train_and_diagnostic(self):
        cfg=replace(self.cfg,geometry=Geometry("pad74"),training=replace(self.cfg.training,steps=2,warmup_steps=0))
        result=train(cfg,k=0)
        measured=padding_diagnostic(result["checkpoint"],limit=0)
        self.assertEqual(measured["results"]["reflect"]["prediction_flip_rate"],0)
        self.assertGreaterEqual(measured["results"]["replicate"]["tile_cosine_distance_mean"],0)

    def test_matrix_resumes_existing_and_starts_missing_runs(self):
        config_path = self.root / "matrix.toml"
        text = "\n".join(f"[{section}]\n" + "\n".join(f"{key} = {json.dumps(value)}" for key,value in values.items())
                         for section,values in self.cfg.to_dict().items())
        config_path.write_text(text, encoding="utf-8")
        output = self.root / "matrix.json"
        with redirect_stdout(io.StringIO()):
            train(self.cfg, seed=0, k=0, stop_after=2)
            main(["matrix", "--config", str(config_path), "--ks", "0,3", "--seeds", "0,1",
                  "--execute", "--resume", "--output", str(output)])
        result = json.loads(output.read_text())
        self.assertTrue(result["executed"])
        self.assertEqual(result["n_runs"], 4)
        self.assertTrue(all(row["train"]["complete"] for row in result["results"]))

    def test_linear_stats_training_only_and_knn(self):
        x=torch.tensor([[0.,0.],[1.,0.],[8.,1.],[9.,1.]])
        y=torch.tensor([0,0,1,1])
        _,mean,std=fit_linear(x,y,2,self.cfg,0,"cpu")
        self.assertTrue(torch.equal(mean,x.mean(0)))
        self.assertEqual(knn_accuracy(torch.eye(2),torch.arange(2),torch.eye(2),torch.arange(2),2,k=1),1)
        chosen=label_subset(y,0.01,0)
        self.assertEqual(len(chosen),2)

    def test_config_rejects_bad_fields(self):
        with self.assertRaises(ValueError): Config.from_dict({"model":{"encodre":"small"}})
        with self.assertRaises(ValueError): Config.from_dict({"geometry":{"mode":"74_no_padding"}})
        with self.assertRaises(ValueError): Config.from_dict({"model":{"relation":"none","relation_weight":1}})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
