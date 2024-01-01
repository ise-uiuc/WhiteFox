
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(10)
    def forward(self, x1):
        v1 = self.batch_norm(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 10, 32, 32)
