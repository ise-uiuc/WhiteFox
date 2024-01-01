
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(1, 5, 2, stride=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(5, 19, 3, stride=2, padding=1, output_padding=1)
        self.conv_t3 = torch.nn.ConvTranspose2d(19, 480, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x0):
        t1 = self.conv_t1(x0)
        t1 = t1 > 0
        t1 = t1 * 0.5
        t1 = torch.where(t1, t1, -t1)
        t2 = self.conv_t2(t1)
        t2 = t2 > 0
        t2 = t2 * 0.5
        t2 = torch.where(t2, t2, -t2)
        t3 = self.conv_t3(t2)
        t3 = t3 > 0
        t3 = t3 * 0.5
        t3 = torch.where(t3, t3, -t3)
        return t3
# Inputs to the model
x0 = torch.randn(8, 1, 14, 16)
