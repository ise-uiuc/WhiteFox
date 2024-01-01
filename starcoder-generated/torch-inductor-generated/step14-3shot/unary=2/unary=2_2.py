
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=8, stride=1, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=4, kernel_size=8, stride=1, padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(in_channels=4, out_channels = 8, kernel_size=8, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = [self.conv_transpose, self.conv_transpose2, self.conv_transpose3][0](x1)
        v2 = self.conv_transpose3(x2)
        v3 = v1+v2+x3
        v4 = torch.relu6(v3)
        v5 = torch.dropout(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
x2 = torch.randn(1, 1, 1, 1)
x3 = torch.randn(1, 1, 1, 1)
