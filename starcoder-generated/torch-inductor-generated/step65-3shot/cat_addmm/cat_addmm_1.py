
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(6, 5, bias=False)
    def forward(self, x):
        x = self.layers(x)
        x = torch.tanh(x)
        x = torch.permute(x, (1, 0, 2))
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.stack((x, x, x, x), dim=2)
        x = torch.sum(x, dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 6)
