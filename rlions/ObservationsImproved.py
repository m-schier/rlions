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

import sys

import jax
import jax.numpy as jnp
import chex

from rlions.Env import EnvState, Program


def make_qubit_positions_from_cells(cells: jax.Array, max_qubits: int) -> jax.Array:
    n_cells, = cells.shape
    return jnp.full(max_qubits, -1, dtype=jnp.int32).at[jnp.where(cells >= 0, cells, max_qubits + 1)].set(jnp.arange(n_cells), mode='drop')


def make_fixed_lookahead(depths, idxs):
    """
    Create a lookahead with fixed positions (padded form) from a lookahead of variable positions (packed form).
    :param depths: Relevant gate depths.
    :param idxs: Variable lookahead of shape (n_cells, lookahead).
    :return: Fixed lookahead of shape (n_cells, lookahead).
    """
    n_cells, lookahead = idxs.shape
    sel_depth = depths[idxs]
    sel_valid = (sel_depth >= 0) & (sel_depth < lookahead)
    return jnp.full_like(idxs, -1).at[
        jnp.arange(n_cells)[..., None], jnp.where(sel_valid, sel_depth, idxs.size + 1)].set(idxs, mode='drop')


def encode_cells_with_program_positions(cells: jax.Array, program: Program, number_encoder, lookahead: int, max_qubits: int,
                                        as_tokens: bool = False, ion_gate_count_number_encoder=None, fixed_depth: bool = False):
    from rlions.Util import find_first_k_gates

    # Assert program shapes
    program_capacity, operand_count = program.operations.shape
    chex.assert_equal(operand_count, 2)
    chex.assert_shape(program.completed, (program_capacity,))

    n_cells, = cells.shape

    positions = make_qubit_positions_from_cells(cells, max_qubits)

    # Make packed lookahead
    lookahead_buffer = find_first_k_gates(program, max_qubits, lookahead)
    chex.assert_shape(lookahead_buffer, (max_qubits, lookahead))

    if fixed_depth:
        # Make padded from packed lookahead
        from rlions.Util import make_gate_depth, make_sub_gate_depth

        # Decide which gate depth approach has better linear complexity
        full_scan_ops = program_capacity
        sub_scan_ops = max_qubits * lookahead

        if full_scan_ops > sub_scan_ops:
            # We can correctly estimate the relevant gate depths from the subset of gates in the packed lookahead buffer
            # rather than the entire program. For long programs this can be much faster.
            depths = make_sub_gate_depth(program, max_qubits, lookahead_buffer)
        else:
            depths = make_gate_depth(program, max_qubits)

        lookahead_buffer = make_fixed_lookahead(depths, lookahead_buffer)
        chex.assert_shape(lookahead_buffer, (max_qubits, lookahead))

    idxs = jnp.where(
        cells[..., None] >= 0,
        lookahead_buffer[cells],
        -1,
    )

    chex.assert_shape(idxs, (n_cells, lookahead))

    # Left broadcast the program index to the (*batch_shape, stack_capacity, lookahead) shape
    valid = ~program.completed[idxs] & (idxs >= 0)

    # Other qubit
    other_qubit = jnp.where(
        program.operations[idxs, 0] == cells[..., None],
        program.operations[idxs, 1],
        program.operations[idxs, 0],
    )

    chex.assert_shape(valid, (n_cells, lookahead))
    chex.assert_shape(other_qubit, (n_cells, lookahead))

    encoded = number_encoder(positions[other_qubit], valid=valid)
    ion_valid = cells >= 0

    entries = [
        ion_valid[..., None],
        encoded.reshape((n_cells, -1)),
    ]

    # Optionally encode the remaining gates directly which should improve observability if lookahead filled
    assert ion_gate_count_number_encoder is None

    tokens = jnp.concatenate(entries, axis=-1)

    if as_tokens:
        return tokens
    else:
        return tokens.reshape(-1)


def make_encoder3(adapter, lookahead: int,
                  max_ion_gate_count: int = 1_000, number_encoder="sinusoidal",
                  steps_max: float = 10_000., steps_bands: int = 4, observe_steps: bool = True,
                  op_bands: int = 4, ion_gate_bands: int = 4,
                  encode_ion_gate_counts: bool = False, fixed_depth: bool = False):
    from rlions.Observations import make_sinusoidal_encoder, make_linear_encoder

    print("Using improved encoder", file=sys.stderr)

    if "sinusoidal" in number_encoder:
        pos_number_encoder = make_sinusoidal_encoder(adapter.n_cells, op_bands, mask_invalid=True, invalid_neutral=True, clip=True)
        ion_gate_number_encoder = make_sinusoidal_encoder(max_ion_gate_count, ion_gate_bands, mask_invalid=True, invalid_neutral=True, clip=True)
        step_number_encoder = make_sinusoidal_encoder(steps_max, steps_bands, mask_invalid=True, invalid_neutral=True, clip=True)
    elif number_encoder == "linear":
        pos_number_encoder = make_linear_encoder(1)
        ion_gate_number_encoder = make_linear_encoder(1)
        step_number_encoder = make_linear_encoder(1)
    else:
        raise ValueError(f"{number_encoder = }")

    s_igc_enc = ion_gate_number_encoder if encode_ion_gate_counts else None

    def nonbatched_encoder(env: EnvState):
        max_qubits = adapter.max_qubits
        cells = adapter.encode(env)
        assert cells.shape[-1] == adapter.n_cells

        entries = [
            encode_cells_with_program_positions(cells, env.program, pos_number_encoder, lookahead, max_qubits, as_tokens=False, ion_gate_count_number_encoder=s_igc_enc, fixed_depth=fixed_depth),
            ion_gate_number_encoder(jnp.sum(~env.program.completed, axis=-1, keepdims=True)),
        ]

        if observe_steps:
            # Formally required to be observable due to us not handling truncations
            # Unless termination through time are random
            entries.append(step_number_encoder(jnp.asarray(env.steps)[..., None]))

        return jnp.concatenate(entries, axis=-1)

    def encoder(state: EnvState):
        fn = nonbatched_encoder

        for _ in jnp.shape(state.steps):
            fn = jax.vmap(fn)

        return fn(state)

    return encoder


def make_encoder_basic(adapter, lookahead: int,
                  max_ion_gate_count: int = 1_000, number_encoder="sinusoidal",
                  steps_max: float = 10_000., steps_bands: int = 4, observe_steps: bool = True,
                  op_bands: int = 4, ion_gate_bands: int = 4,
                  encode_ion_gate_counts: bool = False):
    from rlions.Observations import make_sinusoidal_encoder, make_linear_encoder

    assert not encode_ion_gate_counts

    if "sinusoidal" in number_encoder:
        pos_number_encoder = make_sinusoidal_encoder(adapter.n_cells, op_bands, mask_invalid=True, invalid_neutral=True, clip=True)
        ion_gate_number_encoder = make_sinusoidal_encoder(max_ion_gate_count, ion_gate_bands, mask_invalid=True, invalid_neutral=True, clip=True)
        step_number_encoder = make_sinusoidal_encoder(steps_max, steps_bands, mask_invalid=True, invalid_neutral=True, clip=True)
    elif number_encoder == "linear":
        pos_number_encoder = make_linear_encoder(1)
        ion_gate_number_encoder = make_linear_encoder(1)
        step_number_encoder = make_linear_encoder(1)
    else:
        raise ValueError(f"{number_encoder = }")

    equivalent_length = adapter.max_qubits * lookahead

    def nonbatched_encoder(env: EnvState):
        cells = adapter.encode(env)
        assert cells.shape[-1] == adapter.n_cells

        # Encode cells the basic way
        cells_encoding = pos_number_encoder(cells, valid=cells >= 0).reshape(-1)

        n_prog, = env.program.completed.shape
        if n_prog >= equivalent_length:
            # No need to do any padding is the program is long enough for sub-indexing
            padded_program_completed = env.program.completed
            all_idxs, = jnp.indices((n_prog,))
        else:
            # Program too short, we pad the completed array to the minimum required length and generate indices for that
            # length.
            # Slightly confusing, invalid entries are considered completed
            padded_program_completed = jnp.ones((equivalent_length,), dtype=jnp.bool).at[:n_prog].set(env.program.completed)
            all_idxs, = jnp.indices((equivalent_length,))

        # The first out of bounds index
        idx_max = max(n_prog, equivalent_length)

        # Generate indices. Where the padded program is completed, this is invalid and receives an oob high index.
        # Otherwise, use the running index of that position. Then do an argsort ascending on the result.
        circuit_idxs = jnp.argsort(jax.lax.select(
            padded_program_completed,
            jnp.broadcast_to(idx_max, padded_program_completed.shape),
            all_idxs), descending=False)[:equivalent_length]

        circuit_valid = ~padded_program_completed[circuit_idxs]
        circuit_op1 = env.program.operations[circuit_idxs, 0]
        circuit_op2 = env.program.operations[circuit_idxs, 1]

        op1_encoding = ion_gate_number_encoder(circuit_op1, valid=circuit_valid).reshape(-1)
        op2_encoding = ion_gate_number_encoder(circuit_op2, valid=circuit_valid).reshape(-1)

        total_gate_encoding = ion_gate_number_encoder(jnp.sum(~env.program.completed, axis=-1, keepdims=True)).reshape(-1)

        all_encodings = [cells_encoding, op1_encoding, op2_encoding, total_gate_encoding]

        if observe_steps:
            all_encodings.append(step_number_encoder(jnp.asarray(env.steps)[..., None]).reshape(-1))

        return jnp.concatenate(all_encodings, axis=-1)

    def encoder(state: EnvState):
        fn = nonbatched_encoder

        for _ in jnp.shape(state.steps):
            fn = jax.vmap(fn)

        return fn(state)

    return encoder
