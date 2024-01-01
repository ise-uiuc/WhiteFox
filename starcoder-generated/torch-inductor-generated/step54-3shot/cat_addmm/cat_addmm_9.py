
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
        self.norm = nn.InstanceNorm1d(4, affine=False)
    def forward(self, x):
        x = self.layers(x)
        x = self.norm(x, affine=True)
        return x
# Inputs to the model
x = torch.randn(4, 4)
