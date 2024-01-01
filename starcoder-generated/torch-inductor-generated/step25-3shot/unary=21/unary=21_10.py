
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=1)
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1, padding=0)
    def forward(self, x):
        x1 = self.conv2d(x)
        x2 = torch.tanh(x1)
        x3 = self.conv1d(x2)
        x4 = torch.tanh(x3)
        return x4
# Inputs to the model
x = torch.randn(1, 1, 100)
