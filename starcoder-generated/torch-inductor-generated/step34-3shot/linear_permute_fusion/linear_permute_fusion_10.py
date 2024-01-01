
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(1, 1, 2, 2, 1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t3 = t1.permute(0, 2, 3, 1)
        return self.linear(t3)
# Inputs to the model
x1 = torch.randn(3, 1, 2, 2)
