
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 4, 2, groups=3)
    def forward(self, x):
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 4)
