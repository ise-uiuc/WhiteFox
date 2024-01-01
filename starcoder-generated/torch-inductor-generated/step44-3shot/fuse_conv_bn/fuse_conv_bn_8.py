
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, bias=True)
        self.conv2 = torch.nn.Conv2d(8, 8, 2, bias=True)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, bias=True)
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(self.relu, torch.nn.Conv2d(8, 1, 1, bias=True))
    def forward(self, x2):
        s1 = self.conv1(x2)
        s2 = self.relu(s1)
        s3 = self.conv2(s2)
        s4 = self.relu(s3)
        s5 = self.conv3(s4)
        s6 = self.relu(s5)
        s7 = self.layer(s6)
        return s2
# Inputs to the model
x2 = torch.randn(1, 3, 7, 7)
