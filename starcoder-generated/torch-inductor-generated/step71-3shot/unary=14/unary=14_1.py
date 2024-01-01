
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(11, 1, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv3(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 11, 224, 224)
