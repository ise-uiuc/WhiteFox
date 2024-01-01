
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers_1(x)
        x = self.layers_2(x)
        x = torch.stack((x, x, x, x), dim=2)
        x = x.permute((2, 1, 0))
        x = x.flatten(end_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
