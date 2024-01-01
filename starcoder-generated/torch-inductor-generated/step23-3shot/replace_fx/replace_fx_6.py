
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 50, 3)
    def forward(self, x1):
        x2 = self.conv(x1) ** 1
        x3 = torch.nn.functional.dropout(x2)
        x4 = torch.rand_like(x3)
        x5 = torch.nn.functional.dropout(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 20, 20, 20)
