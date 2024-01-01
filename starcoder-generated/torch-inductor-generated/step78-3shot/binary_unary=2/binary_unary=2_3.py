
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.t3 = torch.tensor(0.5)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 - self.t3
        t3 = F.relu(t2)
        t4 = torch.squeeze(t3, 0)
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
