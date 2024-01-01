
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_1 = torch.nn.ConvTranspose2d(6, 32, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0))
        self.bath = torch.nn.BatchNorm2d(32)
        self.acti = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_t_1(x1)
        v2 = self.bath(v1)
        v3 = self.acti(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 319, 507)
