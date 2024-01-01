
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(2, 4)
        self.layers2 = nn.Linear(4, 6)
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = torch.stack((x, x), dim=1)
        x = x.view(x.shape[0], 6)
        return x
# Inputs to the model
x = torch.randn(2, 2)
