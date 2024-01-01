
class Model(torch.nn.Module):
    def __init__():
        super().__init__()
        self.bn1 = torch.nn.utils.spectral_norm(torch.nn.BatchNorm1d(2600009))
    def forward(self, x1):
        x2 = self.bn1(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2600009)
