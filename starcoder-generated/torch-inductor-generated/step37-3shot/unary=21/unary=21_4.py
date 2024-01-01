
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x, x5):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2 * x5
# Inputs to the model
x = torch.randn(32, 32, 3, 3)
x5 = torch.randn(32, 3, 1, 1)
