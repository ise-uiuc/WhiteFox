
class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Linear(8, 4),
            torch.nn.Linear(4, 2),
            torch.nn.Tanh()
        )
        self.v = torch.nn.Parameter(torch.zeros(8, 2))

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x.matmul(self.v)
        x = x.matmul(self.model[0].weight.transpose(0, 1))
        x = x.matmul(self.model[1].weight.transpose(0, 1))
        x = x.matmul(self.model[2].weight.transpose(0, 1))
        x = x.matmul(self.model[3].weight.transpose(0, 1))
        x = x.matmul(self.model[0].bias)
        x = x.matmul(self.model[1].bias)
        x = x.matmul(self.model[2].bias)
        x = x.matmul(self.model[3].bias)
        return x

def matmul_permute_fusion(module):
    graph = module.graph
    model = module

    pattern_matmul_linear_relu = \
        lambda x: \
        isinstance(x, torch.nn.GraphModule) and \
        (isinstance(x.forward, torch.nn.Sequential) or \
         isinstance(x.forward, torch.nn.ModuleList)) and \
        (len(x.forward) >= 3) and \
        isinstance(x.forward[0], torch.nn.Linear) and \
        isinstance(x.forward[1], torch.nn.ReLU) and \
        isinstance(x.forward[2], torch.nn.Linear)
    pattern_permute_matmul_relu = \
        lambda x: \
        isinstance(x, torch.nn.GraphModule) and \
        (isinstance(x.forward, torch.nn.Sequential) or \
         isinstance(x.forward, torch.nn.ModuleList)) and \
        (len(x.forward) >= 2) and \
        isinstance(x.forward[0], torch.nn.MatMul) and \
        isinstance(x.forward[0], torch.nn.ReLU)

    for node in graph.nodes:
        if node.op == 'call_function' and node.target in [torch.matmul, torch.bmm] and \
            node.has_missing_input_sizes():
            all_users = node.users + node.parent.users
            if pattern_matmul_linear_relu(next(all_users)):
                root_node = next(all_users)
            elif pattern_permute_matmul_relu(next(all_users)):
                permute_node = next(all_users)
                # remove permute node which has no other output nodes
                if len(permute_node.users) == 0 and \
                   len(permute_node.parent.users) == 0:
                    root_node = permute_node.parent
                else:
                    root_node = permute_node
            else:
                continue

            # fuse linear model and matmul node if no other output nodes
            if len(root_node.users) == 0 and \
               len(root_node.parent.users) == 0:
                bias = matmul_linear_fusion(root_node)
                if bias!= None:
                    return
            else:
                matmul_linear_fusion(root_node)

        graph.erase_node(node)

