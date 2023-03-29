# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Sequence

import numpy as np
import pytest

import cirq
from cirq.sim import simulation_state


class DummyQuantumState(cirq.QuantumStateRepresentation):
    def copy(self, deep_copy_buffers=True):
        pass

    def measure(self, axes, seed=None):
        return [5, 3]

    def reindex(self, axes):
        return self


class DummySimulationState(cirq.SimulationState):
    def __init__(self):
        super().__init__(state=DummyQuantumState(), qubits=cirq.LineQubit.range(2))

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> bool:
        return True


def test_measurements():
    args = DummySimulationState()
    args.measure([cirq.LineQubit(0)], "test", [False], {})
    assert args.log_of_measurement_results["test"] == [5]


def test_decompose():
    class Composite(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            yield cirq.X(*qubits)

    args = DummySimulationState()
    assert simulation_state.strat_act_on_from_apply_decompose(
        Composite(), args, [cirq.LineQubit(0)]
    )


def test_mapping():
    args = DummySimulationState()
    assert list(iter(args)) == cirq.LineQubit.range(2)
    r1 = args[cirq.LineQubit(0)]
    assert args is r1
    with pytest.raises(IndexError):
        _ = args[cirq.LineQubit(2)]


def test_swap_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = DummySimulationState()
    with pytest.raises(ValueError, match='Cannot swap different dimensions'):
        args.swap(q0, q1)


def test_rename_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = DummySimulationState()
    with pytest.raises(ValueError, match='Cannot rename to different dimensions'):
        args.rename(q0, q1)


def test_transpose_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    args = DummySimulationState()
    assert args.transpose_to_qubit_order((q1, q0)).qubits == (q1, q0)
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q2))
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q1, q1))


def test_field_getters():
    args = DummySimulationState()
    assert args.prng is np.random
    assert args.qubit_map == {q: i for i, q in enumerate(cirq.LineQubit.range(2))}


@pytest.mark.parametrize('exp', list(range(-4, 5)))
@pytest.mark.parametrize('dim', list(range(1, 6)))
@pytest.mark.parametrize('resolve', [cirq.final_state_vector, cirq.final_density_matrix])
def test_ancilla(exp, dim, resolve):
    class AncillaX(cirq.Gate):
        def _qid_shape_(self):
            return (dim,)

        def _decompose_(self, qubits):
            from cirq.transformers.measurement_transformers import _ModAdd
            ancilla = cirq.NamedQid('Ancilla', dimension=dim)
            yield cirq.XPowGate(exponent=exp, dimension=dim).on(ancilla)
            yield _ModAdd(dimension=dim).on(ancilla, qubits[0])
            yield cirq.XPowGate(exponent=-exp, dimension=dim).on(ancilla)

    q = cirq.LineQid(0, dimension=dim)
    test_circuit = cirq.Circuit(AncillaX().on(q))
    control_circuit = cirq.Circuit(cirq.XPowGate(exponent=exp, dimension=dim).on(q))

    assert np.allclose(resolve(test_circuit), resolve(control_circuit))

@pytest.mark.parametrize('state_type', [cirq.StateVectorSimulationState, cirq.DensityMatrixSimulationState])
def test_basic(state_type):
    from cirq.transformers.measurement_transformers import _MeasurementQid
    q0, q1 = cirq.LineQubit.range(2)
    state = state_type(qubits=[q0, q1])
    state._deferred_mode = True
    circuit = [
        cirq.X(q0),
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls('a'),
    ]
    print()
    for op in circuit:
        cirq.act_on(op, state)
        print(state)
    state._deferred_mode = False

    cirq.act_on(cirq.measure(q1, key='b'), state)
    print(state)

    q_ma = _MeasurementQid('a', q0)
    control = [
        cirq.X(q0),
        cirq.CX(q0, q_ma),
        cirq.CX(q_ma, q1),
        cirq.measure(q_ma, key='a'),
        cirq.measure(q1, key='b'),
    ]
    print()
    state = state_type(qubits=[q0, q1, q_ma])
    for op in control:
        cirq.act_on(op, state)
        print(state)
