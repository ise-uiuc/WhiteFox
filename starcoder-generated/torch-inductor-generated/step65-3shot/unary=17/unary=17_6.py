
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(8, 4, stride=4, padding=0)
    def forward(self, x1):
        # Use hardtanh with -1 and 1 to clip values outside the range
        v1 = torch.tanh(x1)
        v2 = torch.tanh(v1)
        v3 = torch.tanh(v2)
        return v3
# Input to the model
x1 = torch.randn(1, 8, 128)
