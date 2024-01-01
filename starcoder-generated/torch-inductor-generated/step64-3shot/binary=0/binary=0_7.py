
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(4, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 64, stride=1, padding=1)
