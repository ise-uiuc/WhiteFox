
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        res = torch.stack((x, x), dim=0)
        res = res.flatten(0, 2)
        return res
# Inputs to the model
x = torch.randn(2, 2)
