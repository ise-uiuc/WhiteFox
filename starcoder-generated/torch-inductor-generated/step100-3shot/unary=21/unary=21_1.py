
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(16, 64, 3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(32, 16, 64)
