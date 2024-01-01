
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(480, 480, 2, stride=2)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 0.34
        t3 = t2 * 0.68
        t4 = t3 + 0.32
        t5 = torch.sigmoid(t4)
        t6 = t5 + 0.69
        t7 = torch.relu(t6)
        t8 = t7 + 0.45
        t9 = torch.tanh(t8)
        t10 = t9 * 0.26
        return t10
# Inputs to the model
x1 = torch.randn(16, 480, 16, 16)
