
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=128, bias=False, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.rand(2, 16, 256, 256)
