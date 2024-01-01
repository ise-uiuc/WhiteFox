
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(9, 2, 2, stride=(2, 1), padding=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn([1, 9, 5, 5])
