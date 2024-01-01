
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = torch.squeeze(x, dim=2)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 1)
