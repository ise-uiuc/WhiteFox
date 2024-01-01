
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 1, 3, padding=1)
    def forward(self, x_t):
        t1 = self.conv1(x_t)
        t2 = torch.sigmoid(t1)
        t3 = self.conv2(t2)
        t4 = torch.sigmoid(t3)
        t5 = self.conv3(t4)
        t6 = torch.sigmoid(t5)
        t7 = self.conv4(t6)
        t8 = torch.sigmoid(t7)
        return t8
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
