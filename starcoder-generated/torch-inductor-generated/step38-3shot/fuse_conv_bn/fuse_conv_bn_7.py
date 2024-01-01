
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 20, kernel_size=(5, 5), stride=(2, 2), bias=False)
        self.batch_norm_0 = torch.nn.BatchNorm2d(20, affine=True)
        self.conv_1 = torch.nn.Conv2d(20, 40, kernel_size=(5, 5), stride=(1, 1), bias=False)
        self.batch_norm_1 = torch.nn.BatchNorm2d(40, affine=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm_0(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        return x
# Inputs to the model
x = torch.randn(1, 20, 6, 6)
