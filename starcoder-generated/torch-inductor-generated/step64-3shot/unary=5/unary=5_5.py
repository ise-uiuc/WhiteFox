
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 1, 2, stride=1, padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(1, 1, 4, stride=4, padding=0)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(1, 1, 8, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1) * 0.5
        v2 = self.conv_transpose2(x1) * 0.5
        v3 = self.conv_transpose3(x1) * 0.5
        v4 = self.conv_transpose4(x1) * 0.5
        return v1 + v2 + v3 + v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
