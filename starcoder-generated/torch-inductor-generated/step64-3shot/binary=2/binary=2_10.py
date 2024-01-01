
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 3, 8, stride=8)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 12.4
        return v2
# Inputs to the model
x = torch.randn(1, 8, 256)
