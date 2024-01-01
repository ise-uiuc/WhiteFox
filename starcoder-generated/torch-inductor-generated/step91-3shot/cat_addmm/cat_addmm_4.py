
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(4, 4)
