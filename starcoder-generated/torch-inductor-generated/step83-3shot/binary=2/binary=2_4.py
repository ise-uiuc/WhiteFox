
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(9, 23, 1, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 0.1
        return v2
# Inputs to the model
x2 = torch.randn(1, 9, 12)
