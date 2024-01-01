
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.relu6(x1)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x1 = torch.zeros(2, 3, 64, 64)
