
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 1, 3, stride=1, dilation=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 5, 5)
# Outputs from the model
y1 = torch.tensor([[[[-0.5903],
                   [-0.1606],
                   [-0.2212]]]])
