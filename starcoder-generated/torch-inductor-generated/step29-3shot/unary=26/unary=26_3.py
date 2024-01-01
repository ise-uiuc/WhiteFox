
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(10, 20, 3, stride=2, padding=1, output_padding=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(20, 30, 3, stride=2, padding=1, output_padding=1)
        self.conv_t3 = torch.nn.ConvTranspose2d(30, 10, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x4):
        t1 = self.conv_t1(x4)
        t2 = self.conv_t2(t1)
        t3 = self.conv_t3(t2)
        return t3
# Inputs to the model
x4 = torch.randn(6, 10, 8, 8)
