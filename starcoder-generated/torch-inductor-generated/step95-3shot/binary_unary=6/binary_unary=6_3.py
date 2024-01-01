
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, other):
        v1 = torch.nn.functional.linear(x1, torch.randn(x1.shape[1], 1))
        v2 = v1 - other
        v3 = F.relu(v2)
        return v3

# Initializing the model
a1 = torch.randn(5, 3)
m = Model()

# Inputs to the model
other = torch.randn(5, 1)
