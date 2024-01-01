
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(1, 1, 3, stride=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(1, 1, 3, stride=2)
        self.conv_t3 = torch.nn.ConvTranspose2d(1, 1, 3, stride=2)
        self.conv_t4 = torch.nn.ConvTranspose2d(1, 1, 3, stride=3)
    def forward(self, x0):
        t1 = self.conv_t1(x0)
        t2 = self.conv_t2(t1)
        t3 = self.conv_t3(t2)
        t4 = self.conv_t4(t3)
        return t4
# Inputs to the model
x0 = torch.randn(1, 1, 8, 8)
