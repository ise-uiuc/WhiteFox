
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 192, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(192, 1, 1, stride=1, padding=0)
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        t4 = self.avgpool(t3)
        t5 = self.relu(t4)
        t6 = self.tanh(t5)
        t7 = t6 - 3
        t8 = t6 * 3
        return t7.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 4, 96, 96)
