
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 2, kernel_size=(3, 3), groups=1, bias=False, dilation=(1, 1), padding=(2, 2), stride=(3, 3))
        self.conv1 = torch.nn.ConvTranspose2d(2, 4, kernel_size=(3, 3), groups=1, bias=False, dilation=(1, 1), padding=(2, 2), stride=(3, 3))
        self.conv2 = torch.nn.ConvTranspose2d(4, 8, (2, 2), padding=(1, 1), stride=(4, 4))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 65, 129)
