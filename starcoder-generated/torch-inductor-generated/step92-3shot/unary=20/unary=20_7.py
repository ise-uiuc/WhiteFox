
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True)
        self.conv1 = torch.nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(32, 32, kernel_size=(7, 7), stride=1, padding=(3, 3), bias=True)
        self.conv2 = torch.nn.Conv2d(32, 3, kernel_size=1, stride=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv1(v1)
        v3 = self.conv_t2(v2)
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(2, 64, 89, 90)
