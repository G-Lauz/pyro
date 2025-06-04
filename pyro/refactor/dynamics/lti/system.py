from typing import Dict, Union

import numpy

from pyro.refactor.signal import StateSignal
from pyro.refactor.system import DynamicSystem


class LTISystem(DynamicSystem):
    """
    Linear Time Invariant (LTI) system class.
    """

    def __init__(self, 
                 state_matrix: Union[numpy.ndarray, float],
                 input_matrix: Union[numpy.ndarray, float],
                 output_matrix: Union[numpy.ndarray, float],
                 feedforward_matrix: Union[numpy.ndarray, float],
                 initial_states: Union[numpy.ndarray, float] = None
                 ):
        """
        Initialize the LTI system with state-space matrices.

        :param state_matrix: The state matrix A.                   (state_dim, state_dim)
        :param input_matrix: The input matrix B.                   (state_dim, input_dim)
        :param output_matrix: The output matrix C.                 (output_dim, state_dim)
        :param feedforward_matrix: The feedforward matrix D.       (output_dim, input_dim)
        """
        # Ensure matrices are numpy arrays
        state_matrix = self.ensure_array(state_matrix)
        input_matrix = self.ensure_array(input_matrix)
        output_matrix = self.ensure_array(output_matrix)
        feedforward_matrix = self.ensure_array(feedforward_matrix)

        state_signal = StateSignal(name="x",
                             dim=state_matrix.shape[0],
                             initial_values=initial_states if initial_states is not None else numpy.zeros(state_matrix.shape[0]),
                             lower_bounds=numpy.full(state_matrix.shape[0], -numpy.inf),
                             upper_bounds=numpy.full(state_matrix.shape[0], numpy.inf))

        super().__init__(name="LTI System", state_signal=state_signal)

        # TODO: We could expose those signal as parameters of the constructor
        # so the user can set them up as he wants

        self.add_input_port(name="u")
        self.add_output_port(name="y")

        self.state_matrix = state_matrix
        self.input_matrix = input_matrix
        self.output_matrix = output_matrix
        self.feedforward_matrix = feedforward_matrix

    def compute_dynamics(self, time: float, dt: float = 0.01) -> numpy.ndarray:
        """
        Compute the dynamics of the LTI system.

        \equation{begin}
            dx = Ax + Bu
        \equation{end}

        :param states: The state vector of the system.              (state_dim, 1)
        :param inputs: The control input vector of the system.      (input_dim, 1)
        :param dt: The time step of the simulation.                 (float)

        :return: The state derivative vector.                       (state_dim, 1)
        """
        states = self.states
        inputs = self.inputs["u"].values

        dynamics = numpy.dot(self.state_matrix, states) + numpy.dot(self.input_matrix, inputs)
        return dynamics

    def compute_output(self, time: float, dt: float = 0.01) -> Dict[str, numpy.ndarray]:
        """
        Compute the output of the LTI system.

        \equation{begin}
            y = Cx + Du
        \equation{end}

        :param states: The state vector of the system.              (state_dim, 1)
        :param inputs: The control input vector of the system.      (input_dim, 1)
        :param dt: The time step of the simulation.                 (float)

        :return: The output vector.                                 (output_dim, 1)
        """
        states = self.states
        inputs = self.inputs["u"].values

        output = numpy.dot(self.output_matrix, states) + numpy.dot(self.feedforward_matrix, inputs)

        return {
            "y": output
        }

    def ensure_array(self, array: Union[numpy.ndarray, float]) -> numpy.ndarray:
        """
        Ensure the input is a numpy array.

        :param array: The input array.
        :return: The input as a numpy array.
        """
        array = numpy.array(array)

        if array.ndim == 0:
            array = numpy.expand_dims(array, axis=0)

        return array
