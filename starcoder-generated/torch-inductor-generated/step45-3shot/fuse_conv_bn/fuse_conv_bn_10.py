
class Model(torch.nn.Module):
    def __init__(self, inp, mid, out):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = nn.Conv2d(inp, mid, kernel_size=3)
        torch.manual_seed(1)
        self.bn = nn.BatchNorm1d(mid)
        torch.manual_seed(1)
        self.conv2 = nn.Conv1d(mid, out, kernel_size=3)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn(x1)
        x2 = self.conv1(x)
        x2 = self.conv2(x)
        x = x1 + x2
        return x
# Inputs to the model
x = torch.randn(1, 4, 5, 5)
