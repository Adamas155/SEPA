"""Independent tile-local SEPA and a tile-aligned I-JEPA-style full-image control."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import math

import torch
from torch import nn

from .config import Config
from .rng import seed_for


@contextmanager
def initialization(seed, component):
    # Optional heads and different PE sizes cannot perturb other modules' RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for(seed, "initialization", component))
        yield


def position_table(side, dim):
    if dim % 4:
        raise ValueError("2D position width must be divisible by four")
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    omega = 1 / (10000 ** (torch.arange(dim // 4).float() / (dim // 4)))
    def encode(v):
        phase = v.flatten().float()[:, None] * omega[None, :]
        return torch.cat((phase.sin(), phase.cos()), -1)
    return torch.cat((encode(y), encode(x)), -1)


def transformer(dim, depth, heads):
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
    net = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
    # TransformerEncoder clones a layer. Give each cloned layer distinct weights.
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.zeros_(module.in_proj_bias)
    return net


class TileEncoder(nn.Module):
    def __init__(self, dim, depth, heads, tile_size):
        super().__init__()
        if tile_size % 16:
            raise ValueError("tile_size must be divisible by 16; padding belongs in the data transform")
        self.dim, self.tile_size, self.patch_side = dim, tile_size, tile_size // 16
        self.patch_embed = nn.Conv2d(3, dim, 16, 16)
        self.blocks = transformer(dim, depth, heads)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("patch_pe", position_table(self.patch_side, dim))

    def tokens(self, tiles):
        if tiles.ndim != 5 or tuple(tiles.shape[2:]) != (3, self.tile_size, self.tile_size):
            raise ValueError("Expected BxVx3xtile_sizextile_size")
        x = self.patch_embed(tiles.flatten(0, 1)).flatten(2).transpose(1, 2)
        return self.norm(self.blocks(x + self.patch_pe)).reshape(*tiles.shape[:2], -1, self.dim)

    def forward(self, tiles):
        return self.tokens(tiles).mean(2)


class FullImageEncoder(TileEncoder):
    """All visible image patches attend jointly. Hidden pixels are absent before attention.

    A matched tile masking control, not a reproduction of the official I-JEPA
    multiblock masking recipe. Slot pooling yields the same 7/2 predictor interface.
    """
    def __init__(self, dim, depth, heads, tile_size):
        super().__init__(dim, depth, heads, tile_size)
        full_side = 3 * self.patch_side
        full = position_table(full_side, dim).reshape(full_side, full_side, dim)
        pe = [full[r*self.patch_side:(r+1)*self.patch_side,
                   c*self.patch_side:(c+1)*self.patch_side].reshape(-1, dim)
              for r in range(3) for c in range(3)]
        self.register_buffer("global_pe", torch.stack(pe))

    def tokens(self, tiles, observed_slots):
        b, v = tiles.shape[:2]
        if tuple(tiles.shape[2:]) != (3, self.tile_size, self.tile_size) or observed_slots.shape != (b, v):
            raise ValueError("Invalid full-image visible patch inputs")
        patches = self.patch_embed(tiles.flatten(0, 1)).flatten(2).transpose(1, 2)
        patches = patches.reshape(b, v, -1, self.dim)
        x = (patches + self.global_pe[observed_slots]).flatten(1, 2)
        return self.norm(self.blocks(x)).reshape(b, v, -1, self.dim)

    def forward(self, tiles, observed_slots):
        return self.tokens(tiles, observed_slots).mean(2)


class Predictor(nn.Module):
    def __init__(self, encoder_dim, dim, depth, heads):
        super().__init__()
        self.projection = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("slot_pe", position_table(3, dim))
        self.blocks = transformer(dim, depth, heads)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, encoder_dim))

    def forward(self, h, observed_slots, query_slots):
        if h.shape[:2] != observed_slots.shape or query_slots.shape != (len(h), 2):
            raise ValueError("Predictor expects visible slots and exactly two query slots")
        visible = self.projection(h) + self.slot_pe[observed_slots]
        queries = self.mask_token + self.slot_pe[query_slots]
        joint = self.blocks(torch.cat((visible, queries), dim=1))
        return self.output(joint[:, h.shape[1]:])


class RelationHead(nn.Module):
    def __init__(self, dim, hidden, directed=False):
        super().__init__()
        self.directed = directed
        self.net = nn.Sequential(nn.Linear(4 * dim if directed else 2 * dim, hidden),
                                 nn.GELU(), nn.Linear(hidden, 8 if directed else 1))

    def forward(self, h):
        a, b = h[:, :, None, :], h[:, None, :, :]
        if self.directed:
            features = torch.cat((a.expand(-1, -1, h.shape[1], -1),
                                  b.expand(-1, h.shape[1], -1, -1), a-b, a*b), -1)
        else:
            features = torch.cat(((a-b).abs(), a*b), -1)
        return self.net(features)


class SEPA(nn.Module):
    def __init__(self, config: Config, seed=0, method="sepa"):
        super().__init__()
        if method not in {"sepa", "full"}:
            raise ValueError("method must be sepa or full")
        if method == "full" and (config.geometry.mode == "pad74" or config.model.relation != "none"):
            raise ValueError("Full-image reference requires native geometry and no relation head")
        self.method = method
        self.relation_mode = config.model.relation
        dim, depth, heads = config.model.encoder_spec
        cls = FullImageEncoder if method == "full" else TileEncoder
        with initialization(seed, "encoder"):
            self.encoder = cls(dim, depth, heads, config.geometry.tile_size)
        with initialization(seed, "predictor"):
            self.predictor = Predictor(dim, config.model.predictor_dim,
                                       config.model.predictor_depth, config.model.predictor_heads)
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.relation_head = None
        self.aggregation = None
        if self.relation_mode != "none":
            with initialization(seed, "relation_head"):
                self.relation_head = RelationHead(dim, config.model.relation_hidden,
                                                  self.relation_mode == "directed")
        if self.relation_mode == "aggregation":
            with initialization(seed, "aggregation"):
                self.aggregation = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.LayerNorm(dim))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, visible_tiles, observed_slots, query_slots):
        # No canonical identity, true mapping, k, hidden pixels, or teacher targets.
        h = self.encoder(visible_tiles, observed_slots) if self.method == "full" else self.encoder(visible_tiles)
        relations = self.relation_head(h) if self.relation_head is not None else None
        context = h
        if self.aggregation is not None:
            weights = relations.squeeze(-1).sigmoid().detach()
            eye = torch.eye(h.shape[1], device=h.device, dtype=torch.bool)[None]
            weights = weights.masked_fill(eye, 0)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            context = h + self.aggregation(weights @ h)
        return self.predictor(context, observed_slots, query_slots), h, relations

    @torch.no_grad()
    def targets(self, canonical_tiles, query_slots):
        if self.method == "full":
            slots = torch.arange(9, device=canonical_tiles.device).expand(len(canonical_tiles), -1)
            all_targets = self.teacher(canonical_tiles, slots)
        else:
            all_targets = self.teacher(canonical_tiles)
        return all_targets.gather(1, query_slots[..., None].expand(-1, -1, all_targets.shape[-1]))

    def encode(self, canonical_tiles):
        if self.method == "full":
            slots = torch.arange(9, device=canonical_tiles.device).expand(len(canonical_tiles), -1)
            return self.encoder(canonical_tiles, slots)
        return self.encoder(canonical_tiles)

    @torch.no_grad()
    def update_teacher(self, momentum):
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must lie in [0,1]")
        online = dict(self.encoder.named_parameters())
        for name, p in self.teacher.named_parameters():
            p.lerp_(online[name], 1 - momentum)
        buffers = dict(self.encoder.named_buffers())
        for name, b in self.teacher.named_buffers():
            b.copy_(buffers[name])
