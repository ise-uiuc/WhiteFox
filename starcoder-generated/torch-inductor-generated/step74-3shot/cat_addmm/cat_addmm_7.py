
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = torch.transpose(x, 0, 1)
        x = torch.transpose(x, 1, 2)
        x = torch.sum(x, dim=2)
        return x
# Inputs to the model
x = torch.randn(4, 4)
