
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 2, 1, stride=1, padding=1)
    def forward(self, x1):
        v = self.conv(x1)
        return v
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
