
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=15)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x2 = torch.randn(1, 1, 70)
