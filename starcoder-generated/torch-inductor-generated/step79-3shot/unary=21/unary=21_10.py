
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(3,3, kernel_size=19, padding=20, dilation=1)
    def forward(self, t):
        x = self.conv(t)
        y = torch.tanh(x)
        return y
# Inputs to the model
t = torch.randn(1, 3, 1, 41)
