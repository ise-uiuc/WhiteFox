
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        a = self.batch_norm(x1)
        return a
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
