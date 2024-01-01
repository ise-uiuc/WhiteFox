
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 64, 9, stride=1, bias=True, padding=13, dilation=1)
        self.linear = torch.nn.Linear(17656, 17656)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.linear(torch.flatten(v1, 1))
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 12, 4, 4)
