
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout()
        self.dropout2 = torch.nn.Dropout(training=True)
    def forward(self, x1):
        x2 = self.dropout1(x1)
        x3 = self.dropout2(x1)
        return x3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout()
        self.dropout2 = torch.nn.Dropout(training=True)
    def forward(self, x1):
        x2 = self.dropout1(x1)
        return x2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout()
    def forward(self, x1):
        x2 = self.dropout1(x1)
        x3 = self.dropout1(x1, training=True)
        return x3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout()
    def forward(self, x1):
        x2 = self.dropout1(x1, training=True)
        return x2
# Input to the model
x1 = torch.randn(1, 2, 2)

# The code below is for debugging purposes
def check_match(fx_module, replacements):
    # Check if each node with kind `call_function` in fx_module.graph.nodes
    # matches the pattern of a call to a replacement.
    for node in fx_module.graph.nodes:
        if node.kind == "call_function" and node.target in replacements:
            # Make sure the arguments and the kwargs match.
            assert node.args == replacements[node.target]["args"]
            assert getattr(node, "kwargs", None) == replacements[node.target]["kwargs"]

            # Make sure the subgraph matches. The ordering of nodes in the
            # subgraph doesn't matter, but the number and kinds of nodes in
            # the subgraph do need to match.
            assert len(node.args[0].nodes) == len(replacements[node.target]["subgraph"].nodes)
            for other_node in replacements[node.target]["subgraph"].nodes:
                if other_node.name in node.args[0].graph.nodes_map:
                    other_node_match = False
                    other_node_target = node.args[0].graph.nodes_map[other_node.name].target
                    if other_node.op == "placeholder":
                        # If the kind of node is "placeholder", the value is
                        # unknown, and the target matches the name.
                        if other_node_target == other_node.name:
                            other_node_match = True
                    elif other_node.kind == "get_attr" and other_node_target == 'x':
                        # If the kind of node is "get_attr", then the target
                        # should be "x", but the value is unknown.
                        other_node_match = True
                    if other_node_match:
                        continue
                    msg = f"Node {other_node.name} in subgraph at {other_node.op} does not match pattern."
                    msg += f" Other node information: kind: {other_node.kind}, target: {other_node_target}, value: {other_node.value}"
                    msg += f" Pattern node information: kind: {other_node.op}, target: {other_node.target}"
                    raise RuntimeError(msg)
