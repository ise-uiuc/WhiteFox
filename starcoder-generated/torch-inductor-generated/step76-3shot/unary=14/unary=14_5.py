
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(4, 4, 1, stride=1, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(4, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv_transpose_1(x1)
        t2 = torch.sigmoid(t1)
        t3 = t1 * t2
        o1 = self.conv_transpose_2(t3)
        return o1
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
