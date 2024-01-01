
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, kernel_size=(3, 4), stride=(17, 58), padding=(0, 4))
        self.avgpool = torch.nn.AvgPool2d(kernel_size=69, stride=552, padding=44)
    def forward(self, x1):
        #v1 = self.conv_transpose(x1)
        v1 = torch.tanh(v1)

        v2 = self.avgpool(v1)
        v3 = torch.tanh(v2)
        #v4 = self.conv_transpose(v3)
        v4 = torch.tanh(v4)
        v5 = self.avgpool(v4)
        v6 = torch.tanh(v5)
        #v7 = self.conv_transpose(v6)
        v7 = torch.tanh(v7)
        v8 = self.avgpool(v7)
        v9 = torch.tanh(v8)
        #v10 = self.conv_transpose(v9)
        v10 = torch.tanh(v10)

        v11 = self.avgpool(v10)
        v12 = self.conv_transpose(v11)
        #v12 = torch.tanh(v12)
        return v12
# Inputs to the model
x1 = torch.randn(1, 4, 1920, 1080)
