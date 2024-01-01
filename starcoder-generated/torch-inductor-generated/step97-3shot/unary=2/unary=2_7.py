
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, (3, 3), stride=(2, 2), padding=(1, 1))
    def forward(self, x1):
        v1 = torch.tanh(self.conv_transpose(x1))
        v2 = v1 * -0.6678060913085938
        v3 = v2 * -0.5
        v4 = v3 * -0.32288545179367065
        v5 = v4 * -0.7302011585235596
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
