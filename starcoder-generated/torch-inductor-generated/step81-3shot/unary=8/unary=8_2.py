
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v4 = v1 + 3
        v5 = torch.clamp(v4, min=0)
        v6 = torch.clamp(v5, max=6)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 103, 106)
