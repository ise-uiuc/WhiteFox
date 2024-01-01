
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 2, stride=1,  padding=0)
        self.conv2 = torch.nn.Conv2d(3, 20, 1, stride=1,  padding=0)
        self.conv3 = torch.nn.Conv2d(20, 5, 4, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(5, 10, 6, stride=1, padding=0)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(x)
        t3 = self.conv3(t1 + t2)
        t4 = self.conv4(t3)
        return t4
# Inputs to the model
x = torch.randn(1, 10, 256, 256)
