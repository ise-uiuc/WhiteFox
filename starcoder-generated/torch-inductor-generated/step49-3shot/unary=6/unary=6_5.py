
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        t4 = torch.flatten(t3, 1)
        t5 = torch.flatten(t4, 1)
        t6 = torch.flatten(x, 1)
        t7 = torch.flatten(x, 2)
        t8 = torch.flatten(x, 3)
        t9 = torch.flatten(x, -1)
        t10 = torch.flatten(x, -2)
        t11 = torch.flatten(x, 0)
        return (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
