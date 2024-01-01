
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.ConvTranspose1d(3, 4, kernel_size=4, stride=4, padding=2, bias=False)
    def forward(self, x1):
        v1 = self.conv1d(x1)
        v2 = torch.tanh(v1)
        return v2
# Input for the model
x1 = torch.randn(1, 3, 10000)
