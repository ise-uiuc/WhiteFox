
class myModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=(2,2))
        self.relu_1 = torch.nn.ReLU()
        self.conv2d_2 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=(4,4))
        self.conv_4 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv2d_1(x1)
        v2 = self.relu_1(v1)
        v3 = self.conv2d_2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_4(v4)
        #v6 = self.conv_4()
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
