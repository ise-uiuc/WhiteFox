
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 10))
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = self.layers(v1)
        v4 = torch.matmul(v2, v1)
        v5 = torch.matmul(v4, v3)
        return (v2, [v3, v1, v3], v5[0])
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
