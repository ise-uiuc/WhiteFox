
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        negative_slope = 0.2
        t1 = self.conv(x)
        t2 = t1 > 0
        t3 = t1 * negative_slope
        t4 = torch.where(t2, t1, t3)
        return self.relu(t4)
# Inputs to the model
x1 = torch.randn(1, 1, 35, 35)
