
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1000, 1, 8, stride=1, padding=0, bias=False)
    def forward(self, x2):
        h1 = self.conv_t(x2)
        h2 = h1 > 0
        h3 = h1 * 1.648397
        h4 = torch.where(h2, h1, h3)
        return torch.nn.functional.relu6(h4)
# Inputs to the model
x2 = torch.randn(17, 1000, 16, 85)
