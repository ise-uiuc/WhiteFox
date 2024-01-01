
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = 0.3000000119
        v2 = torch.sin(v1)
        v3 = 0.3000000119
        v4 = torch.sin(v3)
        v5 = torch.mul(v2, v4)
        v6 = v5 + 0.900000036
        return v6
# Inputs to the model
x = torch.randn(1, 1, 3)
