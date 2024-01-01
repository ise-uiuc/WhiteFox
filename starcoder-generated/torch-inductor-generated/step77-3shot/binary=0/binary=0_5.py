
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
    def forward(self, x1, other=1):
        var1 = self.conv(x1)
        var2 = var1 + other
        var2 += var1
        return var2
# Inputs to the model
x1 = torch.randn(1, 1, 64)
