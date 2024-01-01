
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 3, 2, stride=1, bias=False)
        # Note: We are using the PyTorch conv2d operator as a conv1d operator because we just need to make conv_transpose and conv2d share parameters in some way
        self.conv2d = torch.nn.Conv2d(3, 3, 2, stride=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv2d(x1)
        v3 = v1 + v2
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 6, 2, 2)
