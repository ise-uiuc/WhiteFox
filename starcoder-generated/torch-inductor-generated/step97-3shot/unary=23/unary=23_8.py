
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bias = np.random.rand(3, 5, 3, 3).astype(dtype=np.float32)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 3, stride=1, padding=1, bias=bias)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
