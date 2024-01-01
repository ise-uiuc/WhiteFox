
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(10, 20, 3, stride=2, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(20, 30, 3, stride=2, bias=False)
        self.conv_t3 = torch.nn.ConvTranspose2d(30, 40, 3, stride=2, bias=False)
        self.conv_t4 = torch.nn.ConvTranspose2d(40, 50, 3, stride=2, bias=False)
    def forward(self, x1):
        x2 = self.conv_t1(x1)
        x3 = self.conv_t2(x2)
        x4 = self.conv_t3(x3)
        x5 = self.conv_t4(x4)
        return x5
# Inputs to the model
x1= torch.randn(32, 10, 128, 128)
