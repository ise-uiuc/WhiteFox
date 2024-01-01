
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(54, 8, 1, stride=1, padding=1) # 54
        self.conv2 = torch.nn.Conv2d(16, 4, 3, stride=2, padding=1) # 16
    def forward(self, x1, x2):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.sigmoid(self.conv2(x2))
        v3 = v1.mul(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 54, 224, 224)
x2 = torch.randn(1, 16, 112, 112)
