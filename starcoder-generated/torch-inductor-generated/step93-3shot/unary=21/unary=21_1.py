
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 6, kernel_size=[3], stride=1, groups=1, bias=False)
    def forward(self, x):
        x1 = self.conv(x)
        x11 = torch.tanh(x1)
        return x11
# Inputs to the model
x = torch.randn(1, 1, 128)
