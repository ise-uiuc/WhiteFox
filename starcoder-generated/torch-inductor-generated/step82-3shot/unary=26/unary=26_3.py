
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 5, (5, 2), stride=(3, 1), padding=(3, 1), bias=False, dilation=2)
    def forward(self, x18):
        x11 = self.conv_t(x18)
        x12 = x11 > 0
        x13 = x11 * -0.066
        x14 = torch.where(x12, x11, x13)
        x15 = x14 * 0.659
        x16 = torch.nn.functional.relu(x15)
        x17 = torch.clamp(x16, max=1.337531)
        return torch.nn.functional.dropout(x17)
# Inputs to the model
x18 = torch.randn(6, 4, 14, 9)
