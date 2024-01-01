
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_normalization = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        v = self.batch_normalization(x)
        return v.tanh()
# Inputs to the model
x1 = torch.randn(1, 1, 1200, 400)
