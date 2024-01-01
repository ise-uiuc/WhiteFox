
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(12, 16, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 - x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 12, 100)
x2 = torch.randn(1, 16, 100)
