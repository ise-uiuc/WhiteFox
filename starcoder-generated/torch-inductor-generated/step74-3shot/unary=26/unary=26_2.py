
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 16, kernel_size=(4, 6), stride=(4, 1), bias=False)
    def forward(self, x6):
        y1 = self.conv_t(x6)
        y2 = y1 > 0
        y3 = y1 * 6.81
        y4 = torch.where(y2, y1, y3)
        return torch.nn.functional.relu(y4)
# Inputs to the model
x6 = torch.randn(8, 1, 14, 28)
