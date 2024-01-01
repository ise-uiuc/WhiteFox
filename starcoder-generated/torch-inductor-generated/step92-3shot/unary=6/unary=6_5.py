
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 64, 1, stride=2, padding=0)
        self.conv_2 = torch.nn.Conv2d(64, 128, 4, stride=4, padding=0)
    def forward(self, x1):
        t1 = self.conv_1(x1)
        t2 = self.conv_2(t1)
        t3 = self.conv_1(t2)
        t4 = self.conv_2(t3)
        t5 = torch.relu_(t2 + t4)
        t6 = t4 + t5
        t7 = t4 + t6
        return t7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
