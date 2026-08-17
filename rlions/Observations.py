#  RLIonS - Reinforcement Learning Ion Shuttling Compiler
#  Copyright (C) 2026 Maximilian Schier, Lea Richtmann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import jax
import jax.numpy as jnp


def make_linear_encoder(max_count: float, clip: bool = False):
    def encode(x, valid=None):
        if clip:
            x = jnp.clip(x, min=0, max=max_count)

        if valid is None:
            return x / max_count
        else:
            return jnp.where(valid, x / max_count, -1.)
    return encode


def make_sinusoidal_encoder(max_count: float, bands: int, mask_invalid: bool = False, invalid_neutral: bool = False,
                            clip: bool = False):
    def encode(x, valid=None):
        if valid is None:
            valid = jnp.ones(x.shape, jnp.bool)

        if clip:
            x = jnp.clip(x, min=0, max=max_count)

        x = x[..., None] / max_count

        *init_batch_dim, init_feature_dim = x.shape
        band_freqs = jnp.pi * (2 ** jnp.linspace(0, bands - 1, bands))[:, None]

        cos_encs = jnp.cos(x[..., None, :] * band_freqs)
        sin_encs = jnp.sin(x[..., None, :] * band_freqs)

        valid_entry = valid * 1. if invalid_neutral else valid * 2. - 1.

        if mask_invalid:
            # Zero out invalid, then concatenate invalid mask
            full_enc = valid[..., None, None] * jnp.concatenate([x[..., None, :], cos_encs, sin_encs], axis=-2)
            full_enc = jnp.concatenate([valid_entry[..., None, None], full_enc], axis=-2)
        else:
            # Just concatenate invalid
            full_enc = jnp.concatenate([valid_entry[..., None, None], x[..., None, :], cos_encs, sin_encs], axis=-2)

        *batch_dim, band_dim, feature_dim = full_enc.shape
        assert tuple(batch_dim) == tuple(init_batch_dim)
        assert feature_dim == init_feature_dim
        assert band_dim == 2 * bands + 2

        return jnp.reshape(full_enc, full_enc.shape[:-3] + (-1,))
        # return jnp.reshape(full_enc, tuple(batch_dim) + (feature_dim * band_dim,))
    return encode
