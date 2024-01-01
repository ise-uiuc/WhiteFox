
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 7, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(7, 7, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x
x = torch.randn(1, 3, 4, 4)
