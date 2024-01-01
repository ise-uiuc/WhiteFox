
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.Conv2d(3, 32, 5, stride=1, padding=0)
    def forward(self, var1):
        v2 = self.t(var1)
        v1 = var1 + v2
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
