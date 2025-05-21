import dataclasses

from typing import Dict, List, Tuple


@dataclasses.dataclass
class Signal:
    name: str
    value: float


class Node:
    name: str

    inputs: Dict[str, Signal]
    outputs: Dict[str, Signal]

    def __init__(self, name: str):
        self.name = name

        self.inputs = {}
        self.outputs = {}

    def add_input(self, signal: Signal):
        self.inputs[signal.name] = signal

    def add_output(self, signal: Signal):
        self.outputs[signal.name] = signal

    def process(self):
        """
        Process the node. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class CompositeNode(Node):
    subnodes: List[Node]
    connections: List[Tuple[Node, str, Node, str]]

    def __init__(self, name: str):
        super().__init__(name)

        self.subnodes = []
        self.connections = []

    def add_subnode(self, node: Node):
        self.subnodes.append(node)

    def connect(self, src: Node, src_port: str, dst: Node, dst_port: str):

        if src is self:
            if src_port not in self.inputs:
                raise KeyError(f"Composite has no input port '{src_port}'")
            signal = self.inputs[src_port]
        else:
            if src_port not in src.outputs:
                raise KeyError(f"{src.name} has no output port '{src_port}'")
            signal = src.outputs[src_port]

        if dst is self:
            self.outputs[dst_port] = signal
        else:
            dst.inputs[dst_port] = signal

        self.connections.append((src, src_port, dst, dst_port))

    def process(self):
        """
        Process all subnodes in the composite node.
        """
        for subnode in self.subnodes:
            subnode.process()

    def check_connections(self):
        """
        Check if there's at least one connection to the inputs and outputs of the composite node.
        """
        # check if subnodes empty
        if not self.subnodes:
            raise ValueError("No subnodes in the composite node.")

        # check if connections empty
        if not self.connections:
            raise ValueError("No connections in the composite node.")

        # check if at least one composite node input/output is connected to a subnode output/output
        inputs_port = set(self.inputs.keys())
        outputs_port = set(self.outputs.keys())
        input_check = False
        output_check = False
        for _, src_port, _, dst_port in self.connections:
            if dst_port in inputs_port:
                input_check = True
            if src_port in outputs_port:
                output_check = True

        if not input_check:
            raise ValueError("No connections to the inputs of the composite node.")

        if not output_check:
            raise ValueError("No connections to the outputs of the composite node.")


class SignalNode(Node):
    def __init__(self, name: str, signal: Signal):
        super().__init__(name)
        self.add_input(signal)
        self.add_output(signal)

    def process(self):
        # No processing needed for a signal node
        pass


if __name__ == "__main__":
    class Adder(Node):
        def process(self):
            a = self.inputs["a"].value
            b = self.inputs["b"].value
            self.outputs["sum"].value = a + b

    # Example usage
    signal_a = Signal("a", 1)
    signal_b = Signal("b", 2)
    signal_sum = Signal("sum", 0)

    node_a = SignalNode("node_a", signal_a)
    node_b = SignalNode("node_b", signal_b)
    adder = Adder("Adder")
    adder.add_input(signal_a)
    adder.add_input(signal_b)
    adder.add_output(signal_sum)

    composite_node = CompositeNode("Composite Node")
    composite_node.add_input(signal_a)
    composite_node.add_input(signal_b)
    composite_node.add_output(signal_sum)

    composite_node.add_subnode(node_a)
    composite_node.add_subnode(node_b)
    composite_node.add_subnode(adder)

    composite_node.connect(node_a, "a", adder, "a")
    composite_node.connect(node_b, "b", adder, "b")
    composite_node.connect(adder, "sum", composite_node, "sum")
    composite_node.connect(composite_node, "a", node_a, "a")
    composite_node.connect(composite_node, "b", node_b, "b")

    # Check connections
    try:
        composite_node.check_connections()
        print("All connections are valid.")
    except ValueError as e:
        print(e)

    print(f"Composite Node: {composite_node.name}")
    print("Subnodes:")
    for subnode in composite_node.subnodes:
        print(f"  - {subnode.name}")
    
    print("Connections:")
    for src, src_port, dst, dst_port in composite_node.connections:
        print(f"  - {src.name}.{src_port} -> {dst.name}.{dst_port}")

    # Process the composite node
    composite_node.process()
    print(f"Signal A: {signal_a.value}")
    print(f"Signal B: {signal_b.value}")
    print(f"Signal Sum: {signal_sum.value}")
