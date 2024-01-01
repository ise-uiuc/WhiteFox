
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.BatchNorm1d(16)
    def forward(self, x):
        x = self.a(x)
        return x
# Inputs to the model
x = torch.randn(2, 16, 1, 1)
