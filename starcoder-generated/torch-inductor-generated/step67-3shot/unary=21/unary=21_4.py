
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 6), bias=False)
        self.tanh_0 = torch.nn.Tanh()
        self.avg_pool_0 = torch.nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5))
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)
        self.tanh_1 = torch.nn.Tanh()
    def forward(self, x0):
        x1 = self.conv_0(x0)
        x2 = self.tanh_0(x1)
        x3 = self.avg_pool_0(x2)
        x4 = self.conv_1(x3)
        x5 = self.tanh_1(x4)
        return x5
# Inputs to the model
x0 = torch.randn(1, 8, 42, 54)
