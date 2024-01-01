
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 6, 1, stride=1, padding=0, output_padding=0)
        self.conv2 = torch.nn.Conv2d(6, 1, 1, stride=1, padding=0, output_padding=0)
        self.convTrans = torch.nn.ConvTranspose2d(1, 64, 1, stride=1, padding=0, output_padding=0)
    def forward(self, x1):  # inputs
        t2 = self.conv1(x1)
        t3 = torch.sigmoid(t2)
        t4 = t2 * t3
        t5 = self.convTrans(t4)
        t6 = self.conv2(t5)
        t7 = torch.sigmoid(t6)
        t8 = t6 * t7
        t9 = self.convTrans(t8)
        t10 = t9 + x1
        return t10
# Inputs to the model
x1 = torch.randn(32,64,16,16)
