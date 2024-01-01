
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 4), nn.Linear(2, 4)])
    def forward(self, x):
        for idx in range(2):
            x = self.layers[idx](x)
            x = torch.cat((x, x, x, x), dim=-1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
