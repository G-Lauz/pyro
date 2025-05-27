import abc

from typing import Dict, List, Tuple, Union

import numpy

from pyro.refactor.signal import Signal
from pyro.refactor.system import System, DynamicSystem


class Model(abc.ABC):
    name: str

    systems: Dict[str, System]

    def __init__(self, name: str):
        self.name = name

        self.systems = {}
        self.ordered_systems = []
        self.connections = {}

    def add_system(self, system: System):
        self.systems[system.name] = system

    def connect(self, src: System, src_port: str, dst: Union[Tuple[System, str], List[Tuple[System, str]]], signal: Signal):
        if isinstance(dst, list):
            for dst_system, dst_port in dst:
                self.connect(src, src_port, (dst_system, dst_port), signal)
            return

        dst_system, dst_port = dst
        src.outputs[src_port] = signal
        dst_system.inputs[dst_port] = signal

        if src.name not in self.connections:
            self.connections[src.name] = {}
        if dst_system.name not in self.connections:
            self.connections[dst_system.name] = {}

        self.connections[src.name][src_port] = (dst_system, dst_port)
        self.connections[dst_system.name][dst_port] = (src, src_port)

    def initialize(self):
        self.ordered_systems = []
        sorted_systems = self.sort_nodes()
        for layer in sorted_systems:
            for system_name in layer:
                system = self.systems[system_name]
                self.ordered_systems.append(system)

    def walk_back(self, system: System, history: List[System] = None):
        if history is None:
            history = []

        for input_port, _ in system.inputs.items():
            port_is_connected = input_port in self.connections[system.name].keys()

            if port_is_connected:
                src_system, _ = self.connections[system.name][input_port]

                if not isinstance(src_system, DynamicSystem):
                    history = self.walk_back(src_system, history)
                    history.append(src_system)
        return history

    def sort_nodes(self):
        # TODO: optimize this function

        ordered_parents = []
        for system in self.systems.values():
            sys_parents = []
            sys_parents = self.walk_back(system, sys_parents)
            sys_parents.append(system)
            ordered_parents.append(sys_parents)

        # merge all lists into a single list
        in_degrees = {}
        for sequence in ordered_parents:
            for i, system in enumerate(sequence):
                if system.name not in in_degrees:
                    in_degrees[system.name] = 0

                if i > 0:
                    in_degrees[system.name] += 1


        sorted_systems = []

        max_length = max([len(sequence) for sequence in ordered_parents])
        for i in range(max_length):
            sys_layer = set()
            neighbors = set()
            for sequence in ordered_parents:
                if i >= len(sequence):
                    continue

                system = sequence[i]
                if in_degrees[system.name] == 0:
                    sys_layer.add(system.name)

                    if i+1 < len(sequence):
                        neighbors.add(sequence[i+1])

            for sys in neighbors:
                in_degrees[sys.name] -= 1

            sorted_systems.append(sys_layer)

        # check for cycles
        remaining_systems = [system for system, in_degree in in_degrees.items() if in_degree > 0]
        if remaining_systems:
            raise ValueError(f"Cycle detected in model {self.name} among subsystems: {remaining_systems}")

        return sorted_systems

    def step(self, time: float = 1.0, dt: float = 0.01):
        for system in self.ordered_systems:
            if isinstance(system, DynamicSystem):
                dynamics = system.compute_dynamics(time=time, dt=dt)
                new_states = system.states.values + dynamics * dt
                system.states.update(new_states)

            output_signals = system.compute_output(time=time, dt=dt)
            system.update(output_signals)

    def update(self, signals: Dict[str, Dict[str, numpy.ndarray]]):
        """
        Update the model with new signals.

        :param signals: The signals to update the model with.         (dict)
        """
        for sys_name, sys_signals in signals.items():
            if sys_name in self.systems:
                self.systems[sys_name].update(sys_signals)
            else:
                raise ValueError(f"System {sys_name} not found in model {self.name}.")
