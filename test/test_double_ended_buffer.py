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
from rlions.Elements.DoubleEndedBuffer import DoubleEndedBuffer, de_buffer_init, de_buffer_can_push, de_buffer_push_front
from rlions.Elements.DoubleEndedBuffer import de_buffer_can_pop_back, de_buffer_can_pop_front, de_buffer_push_back
from rlions.Elements.DoubleEndedBuffer import de_buffer_pop_back, de_buffer_pop_front, de_buffer_rotate_front, de_buffer_rotate_back


def test_no_push_zero_sized():
    rb = de_buffer_init(0)
    assert not de_buffer_can_push(rb)


def test_push_front_correct():
    rb = de_buffer_init(3)
    assert rb.front_count == 0
    assert rb.back_count == 0
    rb = de_buffer_push_front(rb, 1)
    assert rb.front_count == 1
    assert rb.back_count == 0
    assert rb.buffer[:1] == 1
    rb = de_buffer_push_front(rb, 2)
    assert rb.front_count == 2
    assert rb.back_count == 0
    assert jnp.allclose(rb.buffer[:2], jnp.array([2, 1]))


def test_push_back_correct():
    rb = de_buffer_init(3)
    assert rb.front_count == 0
    assert rb.back_count == 0
    rb = de_buffer_push_back(rb, 1)
    assert rb.front_count == 0
    assert rb.back_count == 1
    assert rb.buffer[-1:] == 1
    rb = de_buffer_push_back(rb, 2)
    assert rb.front_count == 0
    assert rb.back_count == 2
    assert jnp.allclose(rb.buffer[-2:], jnp.array([1, 2]))


def test_can_push_front():
    rb = de_buffer_init(3)
    assert de_buffer_can_push(rb)
    rb = de_buffer_push_front(rb, 1)
    assert de_buffer_can_push(rb)
    rb = de_buffer_push_front(rb, 2)
    assert de_buffer_can_push(rb)
    rb = de_buffer_push_front(rb, 3)
    assert not de_buffer_can_push(rb)


def test_can_push_back():
    rb = de_buffer_init(3)
    assert de_buffer_can_push(rb)
    rb = de_buffer_push_front(rb, 1)
    assert de_buffer_can_push(rb)
    rb = de_buffer_push_front(rb, 2)
    assert de_buffer_can_push(rb)
    rb = de_buffer_push_front(rb, 3)
    assert not de_buffer_can_push(rb)


def test_pop_front():
    rb = de_buffer_init(3)
    assert not de_buffer_can_pop_front(rb)
    assert not de_buffer_can_pop_back(rb)
    rb = de_buffer_push_front(rb, 1)
    assert de_buffer_can_pop_front(rb)
    assert not de_buffer_can_pop_back(rb)
    rb = de_buffer_push_front(rb, 2)
    assert de_buffer_can_pop_front(rb)
    assert not de_buffer_can_pop_back(rb)
    rb = de_buffer_push_back(rb, 3)
    assert de_buffer_can_pop_front(rb)
    assert de_buffer_can_pop_back(rb)
    assert not de_buffer_can_push(rb)
    assert jnp.allclose(rb.buffer, jnp.array([2, 1, 3]))
    rb, el = de_buffer_pop_front(rb)
    assert el == 2
    assert rb.front_count == 1
    assert rb.back_count == 1
    rb, el = de_buffer_pop_front(rb)
    assert el == 1
    assert rb.front_count == 0
    assert rb.back_count == 1


def test_pop_back():
    rb = de_buffer_init(3)
    rb = de_buffer_push_back(rb, 1)
    rb = de_buffer_push_back(rb, 2)
    rb = de_buffer_push_back(rb, 3)
    rb, el = de_buffer_pop_back(rb)
    assert rb.front_count == 0
    assert rb.back_count == 2
    assert el == 3
    rb, el = de_buffer_pop_back(rb)
    assert rb.front_count == 0
    assert rb.back_count == 1
    assert el == 2
    rb, el = de_buffer_pop_back(rb)
    assert rb.front_count == 0
    assert rb.back_count == 0
    assert el == 1


def test_rotate_front():
    rb = de_buffer_init(5)
    rb = de_buffer_push_front(rb, 1)
    rb = de_buffer_push_front(rb, 2)
    rb = de_buffer_push_back(rb, 3)
    rb = de_buffer_push_back(rb, 4)
    rb = de_buffer_rotate_front(rb)
    assert rb.front_count == 4
    assert rb.back_count == 0
    assert jnp.allclose(rb.buffer[:4], jnp.array([2, 1, 3, 4]))


def test_rotate_back():
    rb = de_buffer_init(5)
    rb = de_buffer_push_front(rb, 1)
    rb = de_buffer_push_front(rb, 2)
    rb = de_buffer_push_back(rb, 3)
    rb = de_buffer_push_back(rb, 4)
    rb = de_buffer_rotate_back(rb)
    assert rb.front_count == 0
    assert rb.back_count == 4
    assert jnp.allclose(rb.buffer[1:], jnp.array([2, 1, 3, 4]))
