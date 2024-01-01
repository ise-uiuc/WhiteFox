
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch = torch.nn.BatchNorm2d(**kwargs) # kwargs could be empty
    def forward(self, x1):
        v1 = self.batch(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
