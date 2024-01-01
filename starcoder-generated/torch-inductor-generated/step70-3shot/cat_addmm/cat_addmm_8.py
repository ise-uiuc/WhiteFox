
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x)).transpose(dim0=1, dim1=0)
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 1)
