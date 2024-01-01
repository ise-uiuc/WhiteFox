
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 1, stride=2)
        self.conv2 = torch.nn.Conv2d(4, 8, 1, stride=2)
        self.conv3 = torch.nn.Conv2d(16, 32, 1, stride=2)
    def forward(self, s):
        s1 = self.conv1(s)
        s2 = self.conv2(s1)
        s3 = torch.tanh(s2)
        s4 = torch.tanh(s3)
        return self.conv3(s4)
# Inputs to the model
s = torch.randn(1, 1, 256, 256)
