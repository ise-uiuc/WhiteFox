
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(10, 20, 3, padding=1)
        self.conv_t_1 = torch.nn.ConvTranspose2d(20, 20, 3, stride=2)
        self.adaptive_pooling = torch.nn.AdaptiveAvgPool2d((1, None))
        self.conv_t_2 = torch.nn.ConvTranspose2d(20, 20, 3, stride=2, output_padding=1)
        self.conv_2 = torch.nn.Conv2d(20, 20, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_t_1(v1)
        v3 = self.adaptive_pooling(v2)
        v4 = self.conv_t_2(v2)
        v5 = self.conv_2(v3)
        v6 = torch.sigmoid(v4 + v5)
        v7 = torch.tanh(v6)
        return v7
# Inputs to the model
x1 = torch.randn(3, 10, 40, 50)
