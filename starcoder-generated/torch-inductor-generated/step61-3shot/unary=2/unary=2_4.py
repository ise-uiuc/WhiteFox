
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = x
        v2 = torch.transpose(v1, 3, 2)
        v3 = x * 0.5
        v4 = v2 * v3
        v5 = torch.transpose(v4, 3, 2)
        v6 = torch.sum(v5)
        return v6
# Inputs to the model
x1 = torch.randn(3, 4, 5, 6)
