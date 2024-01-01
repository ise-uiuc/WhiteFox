
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 4, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x4):
        v1 = self.conv1(x4)
        v10 = self.conv2(v1)
        return v10
# Inputs to the model
x4 = torch.randn(1, 3, 128, 128)
