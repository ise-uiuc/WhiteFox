
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x3):
        return self.conv3(self.conv2(self.conv1(x3)))
# Inputs to the model
x3 = torch.randn(1, 3, 4, 4)
