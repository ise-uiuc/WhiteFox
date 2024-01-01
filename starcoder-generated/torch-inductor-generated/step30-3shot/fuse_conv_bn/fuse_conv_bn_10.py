
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 3)
        self.conv2 = torch.nn.Conv2d(2, 2, 3)
        self.conv3 = torch.nn.Conv2d(2, 2, 3)
        self.conv4 = torch.nn.Conv2d(2, 2, 3)
    def forward(self, x):
        # Input 1:
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        # Input 2:
        h3 = self.conv3(v1)
        h4 = self.conv4(v2)
        return v2, h4
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
