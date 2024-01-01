
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 32, kernel_size=(5, 15), stride=(2, 1), padding=(2, 10), bias=True)
        self.conv_t2 = torch.nn.ConvTranspose2d(32, 31, kernel_size=(52, 19), stride=(15, 1), padding=(17, 4), bias=True)
        self.conv_t3 = torch.nn.ConvTranspose2d(31, 3, kernel_size=(16, 27), stride=(2, 10), padding=(1, 1), bias=True)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        v3 = self.conv_t3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 270, 430)
