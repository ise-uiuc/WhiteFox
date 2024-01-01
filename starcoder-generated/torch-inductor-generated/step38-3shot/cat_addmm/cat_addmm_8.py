
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(20, 4)
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(-1, 4, 5, 5)
        x = x.permute([3, 0, 1, 2])
        t = (x, x)
        x = torch.flatten(t)
        x = torch.stack((x, x, x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(8, 20)
