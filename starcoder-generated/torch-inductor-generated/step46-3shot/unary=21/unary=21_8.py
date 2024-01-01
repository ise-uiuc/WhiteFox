
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(7, 12),stride=(3,7))
    def forward(self, x):
        x1 = torch.tanh(self.conv(x))
        return x1
# Inputs to the model
x = torch.randn(1, 1, 142, 546)
