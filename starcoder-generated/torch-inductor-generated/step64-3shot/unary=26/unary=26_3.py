
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(1, 19, 13, stride=8, padding=13, output_padding=(5, 10, 4))
    def forward(self, x3):
        v1 = self.conv_t(x3)
        return v1
# Inputs to the model
x3 = torch.randn(27, 1, 48, 73, 54, dtype=torch.float, device='cuda')
