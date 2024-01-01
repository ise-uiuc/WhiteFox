
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=(1, 0), dilation=(2, 1))
        self.conv_transpose.weight = torch.nn.Parameter(torch.tensor([[-0.6177, 0.5829, -0.9640]], dtype=torch.float32))
        self.conv_transpose.bias = torch.nn.Parameter(torch.tensor([[-1.6924]], dtype=torch.float32))
    def forward(self, x1):
        def helper(x):
            v1 = self.conv_transpose(x)
            v2 = torch.clamp_min(v1 + 1, -3)
            return torch.clamp_max(v2, 2) / 11
        v3 = torch.sigmoid(helper(torch.tanh(x1)))
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
