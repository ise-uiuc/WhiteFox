
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 32)
        self.layers_2 = nn.Linear(32, 32)
    def forward(self, x):
        x = self.layers(x)
        x = F.selu(x)
        x = self.layers_2(x)
        x = F.selu(x)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
