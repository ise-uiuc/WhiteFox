
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(15, 10, 2, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 5, 2, stride=2, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(5, 2, 2, stride=2, padding=1)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(2, 3, (3, 6), stride=(3, 6))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = self.conv_transpose4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(3, 15, 6, 3)
