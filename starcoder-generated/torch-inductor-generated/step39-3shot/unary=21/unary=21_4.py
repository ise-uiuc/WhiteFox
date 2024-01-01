
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 20, kernel_size=(1, 3), padding=0, bias=False)
    def forward(self, x26):
        v27 = self.conv(x26)
        v28 = torch.tanh(v27)
        return v28
# Inputs to the model
x26 = torch.randn(3, 15, 129, 1)
