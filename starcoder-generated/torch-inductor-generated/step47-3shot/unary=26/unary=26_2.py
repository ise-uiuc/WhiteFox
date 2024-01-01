
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(105, 277, 5, stride=1, padding=0, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -0.0412
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.softmax(x4, dim=-1)
# Inputs to the model
x = torch.randn(2, 105, 96, 64)
