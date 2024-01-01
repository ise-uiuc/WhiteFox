
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.linear1 = torch.nn.Linear(1280, 1024)
        self.linear2 = torch.nn.Linear(1024, 256)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.linear1(t1)
        t3 = self.linear2(t2)
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
