
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 10))
    def forward(self, x1, x2):
        v1 = x1.permute(2, 0, 1)
        v2 = self.layers(v1)
        v3 = x2.permute(2, 0, 1)
        v4 = torch.matmul(v1, v3)
        return (v1, v2, v4)
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 3, 2)
