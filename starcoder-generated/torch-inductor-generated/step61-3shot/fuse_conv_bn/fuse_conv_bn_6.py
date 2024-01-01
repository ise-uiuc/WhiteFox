
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(16, 16, 3, padding=0)
        torch.manual_seed(0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, groups=4)
        torch.manual_seed(1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, padding=0, groups=4)
    def forward(self, x):
        t = self.conv1(x)
        t1 = self.conv2(t)
        t2 = self.conv3(t)
        return t1 + t2
# Inputs to the model
x1 = torch.randn(1, 16, 5, 5)
