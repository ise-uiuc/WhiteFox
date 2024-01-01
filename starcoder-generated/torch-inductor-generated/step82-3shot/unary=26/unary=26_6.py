
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 3, 9, stride=1, padding=1, output_padding=4)
    def forward(self, x11):
        x1 = self.conv_t(x11)
        x2 = x1 > 0
        x5 = torch.where(x2, x1, -x1)
        x3 = torch.neg(x5)
        x4 = torch.floor(x3)
        return torch.nn.functional.relu(x4)
# Inputs to the model
x11 = torch.randn(1, 5, 1, 4)
