
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 1, 1, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.89
        return v2
# Inputs to the model
x = torch.randn(7, 8, 10)
