
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(7, 7, 1, stride=1, padding=0)
        self.conv_transpose2_1 = torch.nn.ConvTranspose2d(7, 7, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv_transpose2(x1)
        t2 = torch.sigmoid(t1)
        t3 = t1 * t2
        t4 = self.conv_transpose2_1(x1)
        t5 = torch.sigmoid(t4)
        t6 = t3 + t5
        return t6
# Inputs to the model
x1 = torch.randn(1, 7, 64, 64)
