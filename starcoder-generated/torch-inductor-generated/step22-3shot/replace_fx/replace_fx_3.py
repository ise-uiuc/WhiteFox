
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1d1 = torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, groups=1, bias=True, padding=0)
        self.conv1d2 = torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, groups=1, bias=True, padding=0)
    def forward(self, x1):
        x2 = self.conv1d1(x1)
        x3 = self.conv1d2(x2)
        x4 = x3 ** 2
        x5 = x4 + x2
        return x5
# Inputs to the model
x1 = torch.randn(1, 1, 100)
