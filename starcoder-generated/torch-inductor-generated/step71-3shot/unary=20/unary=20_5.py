
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 78, kernel_size=(1, 6), stride=(1, 5), padding=(2, 6))
        self.conv_t_1 = torch.nn.ConvTranspose2d(78, 140, kernel_size=(3, 7), stride=(2, 8), padding=(1, 9))
        self.conv_t_2 = torch.nn.ConvTranspose2d(162, 7, kernel_size=(13, 9), stride=(3, 7), padding=(9, 7), bias=True)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.cat((v2, x1), 1)
        v4 = self.conv_t_1(v3)
        v5 = self.conv_t_2(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 100, 160)
