
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        v1 = self.conv_t1(x)
        v2 = self.conv_t2(v1)
        v3 = self.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 256, 256)
