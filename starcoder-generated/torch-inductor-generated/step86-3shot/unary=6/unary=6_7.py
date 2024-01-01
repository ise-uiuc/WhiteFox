
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(1, 1, 2, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1, x2):
        t1 = self.relu(x1)
        t2 = self.conv(t1)
        t3 = self.sigmiod(x2)
        t4 = t2 * t3
        t5 = torch.clamp_max(t4)
        return t4
# Inputs to the model
x1 = torch.randn(5, 5, 5, 5)
x2 = torch.randn(5, 5, 5, 5)
