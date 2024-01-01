
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.clamp1 = torch.nn.ReLU6()
        self.clamp2 = torch.nn.ReLU6()
        self.div1 = torch.nn.Dropout()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = self.clamp1(v2)
        v4 = self.clamp2(v3)
        v5 = self.div1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
