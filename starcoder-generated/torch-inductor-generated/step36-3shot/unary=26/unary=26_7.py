
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(162, 2, kernel_size=(3, 9), stride=(2, 5), bias=False)
    def forward(self, x8):
        x1 = self.conv_t(x8)
        x2 = x1 > 0
        x3 = x1 * -3.88
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.relu(x4)
# Inputs to the model
x8 = torch.randn(11, 162, 10, 31)
