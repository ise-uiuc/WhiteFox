
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.transpose(v1, 1, 2)
        v3 = v2.reshape(1, 30400, 1024)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 513, 999)
