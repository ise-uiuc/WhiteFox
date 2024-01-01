
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(160, 408, 3, padding=0, bias=False)
    def forward(self, x4):
        b1 = self.conv_t(x4)
        b3 = b1 * 14.95
        b4 = torch.nn.functional.dropout(torch.nn.functional.hardtanh(b3, -3, 3), 0, False)
        return b4
# Inputs to the model
x4 = torch.randn(47, 160, 17, 25)
