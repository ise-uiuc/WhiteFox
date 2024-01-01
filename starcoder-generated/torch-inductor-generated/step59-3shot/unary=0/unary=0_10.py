
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(6)
    def forward(self, x4):
        v1 = self.batch_norm(x4)
        return v1
# Inputs to the model
x4 = torch.randn(9, 6, 85, 9)
