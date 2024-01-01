
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 32, kernel_size=(4, 15), stride=1, padding=(1, 10), bias=True)
        self.conv_t2 = torch.nn.ConvTranspose2d(32, 32, kernel_size=(5, 15), stride=1, padding=(2, 10), bias=True)
        self.conv_t3 = torch.nn.ConvTranspose2d(32, 32, kernel_size=(6, 15), stride=1, padding=(3, 10), bias=True)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        v3 = self.conv_t3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 65, 90)
