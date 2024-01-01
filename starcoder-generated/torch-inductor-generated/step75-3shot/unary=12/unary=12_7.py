
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Conv2d(3, 32, 8, stride=4, padding=4)
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.mul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
