
class Mobilenet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.depthwise_conv_transpose = torch.nn.ConvTranspose2d(4, 4, kernel_size=5, stride=5, padding=0, groups=4, bias=False)
        self.batch_norm2 = torch.nn.BatchNorm2d(4)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 16, kernel_size=4, stride=4, padding=0, groups=4, bias=False)
    def forward(self, x0):
        v0 = x0.to(torch.float32)
        v1 = 0.20134112556928627 * v0
        v2 = self.conv_transpose(v1)
        v3 = self.batch_norm(v2)
        v4 = torch.tanh(v3)
        v5 = self.batch_norm(v4)
        v6 = 0.7251560127137529 * v5
        v7 = 0.18161727410737773 * v6
        v8 = self.depthwise_conv_transpose(v7)
        v9 = self.batch_norm2(v8)
        v10 = torch.tanh(v9)
        v11 = self.batch_norm2(v10)
        v12 = 0.13360640258776571 * v11
        v13 = 0.03194726364625066 * v0
        v14 = self.conv_transpose2(v13)
        return v14
v0 = torch.randn(3, 1, 56, 56)
model = Mobilenet()
model(v0)
