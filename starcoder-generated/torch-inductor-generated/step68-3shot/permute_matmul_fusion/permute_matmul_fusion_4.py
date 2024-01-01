
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 10))
    def forward(self, x):
        v1 = x.permute(0, 2, 1)
        v2 = self.layers(v1)
        return v1
# Inputs to the model
x = torch.randn(1, 2, 2)
