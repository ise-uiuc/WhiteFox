
class ModelLogSoftmax(torch.nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=)
    def forward(self, x1):
        y1 = self.conv(x1)
        l1 = self.logsoftmax(l1)
        return l1
# Inputs to the model
x1 = torch.randn(1, 3, 12, 12)
