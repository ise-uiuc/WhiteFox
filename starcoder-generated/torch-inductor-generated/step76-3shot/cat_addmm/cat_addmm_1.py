
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 1)
        self.bn = nn.BatchNorm1d(1, momentum=0.0, affine=True)
    def forward(self, x):
        x = self.bn(self.layers(x.view(1,3)))
        x = torch.cat([x, x], dim=1)
        return x.view(1,6)
# Inputs to the model
x = torch.randn(3,3)
