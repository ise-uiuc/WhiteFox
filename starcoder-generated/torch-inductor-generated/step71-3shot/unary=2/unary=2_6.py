
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 97, (3, 25), stride=(1, 4), padding=0)
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = torch.nn.Softmax(dim=3)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = self.sigmoid(v1)
        v3 = torch.cat((x2, v2), 1)
        v4 = self.adaptive_avg_pool2d(v3)
        v5 = v4.transpose(3, 2)
        v6 = v5.contiguous()
        v7 = self.softmax(v6)
        return v7
# Inputs to the model
x1 = torch.randn(3, 1, 4, 2)
x2 = torch.randn(3, 97, 1, 1)
