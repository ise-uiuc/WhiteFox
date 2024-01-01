
class Model_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3)
        self.conv2 = torch.nn.Conv2d(4, 5, 3)
    def forward(self, x1):
        s1 = self.conv1(x1)
        s2 = self.conv2(s1)
        return s2
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
