
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(t1)
        t3 = torch.tanh(t2)
        t4 = self.conv1(x)
        t5 = self.conv2(t4)
        t6 = torch.tanh(t5)
        t7 = self.conv1(x)
        t8 = self.conv2(t7)
        t9 = torch.tanh(t8)
        t10 = self.conv1(x)
        t11 = self.conv2(t10)
        t12 = torch.relu(t11)
        return torch.add(t3, t6), torch.add(t9, t12)
# Inputs to the model
x = torch.randn(1, 16, 100, 100)
