
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 2, stride=(3, 2), padding=(5, 1), dilation=(3, 1))
        self.conv2 = torch.nn.Conv2d(1, 7, 2, stride=(3, 2), padding=(5, 1), dilation=(3, 1))
        self.conv3 = torch.nn.Conv2d(6, 5, 2, stride=(3, 2), padding=(5, 1), dilation=(3, 1))
        self.conv4 = torch.nn.Conv2d(3, 1, 2, stride=(3, 2), padding=(5, 1), dilation=(3, 1))
        self.conv5 = torch.nn.Conv2d(2, 6, 2, stride=(3, 2), padding=(5, 1), dilation=(3, 1))
    def forward(self, img):
        v1 = torch.tanh(self.conv(img))
        v2 = torch.tanh(self.conv2(v1))
        v3 = torch.tanh(self.conv3(v2))
        v4 = torch.tanh(self.conv4(img))
        v5 = torch.tanh(self.conv5(v1))
        v6 = torch.tanh(torch.add(v5, v6))
        return v3
# Inputs to the model
img = torch.randn(1, 3, 64, 64)
