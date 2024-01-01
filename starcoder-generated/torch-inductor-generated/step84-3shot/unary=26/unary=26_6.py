
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(392, 626, 7, stride=2, padding=1, bias=False)
    def forward(self, x12):
        x1 = self.conv_t(x12)
        x2 = x1 > 0 
        x3 = x1 * 0.214773
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.leaky_relu(x4)
# Inputs to the model
x12 = torch.randn(1, 392, 8, 31)
