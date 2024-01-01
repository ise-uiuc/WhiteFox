
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = torch.nn.MaxPool2d(2, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(7, 11, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.maxpool1(x1)
        v2 = self.conv1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 18, 16)
