
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Conv2d(3, 3, 1)
        self.layers_2 = nn.Conv2d(3, 3, 1)
    def forward(self, x):
        x = self.layers_1(x)
        x = x.flatten(start_dim=1)
        x = self.layers_2(x)
        x = x.mean(dim=(1, 2))
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
