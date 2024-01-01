
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
 
    def forward(self, x1):
        v0 = torch.nn.functional.gelu(x1)
        v1 = torch.nn.functional.gelu(x1, False)
        v2 = torch.nn.functional.gelu(x1, True)
        l1 = self.linear(x1)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        v3 = torch.nn.functional.leaky_relu(x1)
        v4 = torch.nn.functional.leaky_relu(x1, 0.3933874983975471)
        v5 = torch.nn.functional.leaky_relu(x1, 0.3933874983975471, True)
        v6 = torch.nn.functional.relu(x1)
        v7 = torch.nn.functional.relu(x1, True)
        v8 = torch.nn.functional.mish(x1)
        v9 = torch.nn.functional.hardsigmoid(x1)
        v10 = torch.nn.functional.hardsigmoid(x1, True)
        v11 = torch.nn.functional.hardswish(x1)
        return (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
