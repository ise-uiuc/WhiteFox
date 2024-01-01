
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 3)
    def forward(self, x):
        y = self.conv(x)
        return y
# Inputs to the model
x = torch.randn(1, 1, 4)
