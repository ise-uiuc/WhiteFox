
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(220, 119, 1, stride=1, padding=1)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_3 = torch.nn.ConvTranspose2d(119, 90, 1, stride=1, padding=1)
        self.conv_4 = torch.nn.ConvTranspose2d(90, 79, 1, stride=1, padding=1)
        self.conv_5 = torch.nn.ConvTranspose2d(79, 67, 1, stride=1, padding=1)
        self.conv_6 = torch.nn.ConvTranspose2d(67, 59, 1, stride=1, padding=1)
        self.conv_7 = torch.nn.ConvTranspose2d(59, 55, 1, stride=1, padding=1)
        self.conv_8 = torch.nn.ConvTranspose2d(55, 50, 1, stride=1, padding=1)
        self.max_pool_9 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(50, 50, 2, stride=2, padding=1)
    def forward(self, x10):
        v20 = self.conv_1(x10)
        v21 = self.max_pool_2(v20)
        v22 = self.conv_3(v21)
        v23 = self.conv_4(v22)
        v24 = self.conv_5(v23)
        v25 = self.conv_6(v24)
        v26 = self.conv_7(v25)
        v27 = self.conv_8(v26)
        v28 = self.max_pool_9(v27)
        v29 = self.conv_transpose_10(v28)
        return v29
# Inputs to the model
x10 = torch.randn(1, 220, 5, 19)
