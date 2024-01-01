
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.add
        self.conv1d = torch.nn.Conv1d(2, 6, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv1d(x)
        return v2
# Inputs to the model
x1 = torch.randn(20, 2, 2)
