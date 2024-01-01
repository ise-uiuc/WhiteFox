
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 9, 1, stride=1, padding=1, output_padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(9, 9, 1, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(9, 9, 4, stride=1, padding=1, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv_transpose3(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
