
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(input_channels=2, out_channels=2, kernel_size=2)
    def forward(self, x1):
        x1.expand((1, -1, -1))
        v0 = x1
        v1 = self.conv1d(v0)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 3)
