
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(11)
        self.conv1 = torch.nn.Conv2d(32, 24, 3, stride=1, padding1=1, bias=True)
        self.conv2 = torch.nn.Conv2d(24, 16, 2, stride=2, padding=7, bias=True)
        self.conv3 = torch.nn.Conv2d(16, 8, 5, stride=2, padding2=1, bias=True)
    def forward(self, x):
        return self.conv1(x) + self.conv2(self.conv1(x)) + self.conv3(self.conv2(self.conv1(x))) + torch.relu(self.conv3(self.conv2(self.conv1(x))))
# Inputs to the model
x = torch.randn(1, 32, 224, 224)
