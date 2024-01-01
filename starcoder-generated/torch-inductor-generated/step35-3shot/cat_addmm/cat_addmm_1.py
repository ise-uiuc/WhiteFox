
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        x2 = x.permute(0, 2, 1)
        x = torch.cat([x, x2], dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 2)
