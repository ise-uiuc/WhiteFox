
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 1), stride=(1, 1), padding=0, dilation=1)
        self.conv4_2 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 4), stride=(1, 1), padding=(0, 1), dilation=1)
        self.conv7_1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1), dilation=1, bias=True)
        self.conv8_1 = torch.nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(4, 4), stride=(2, 2), padding=1, dilation=1, bias=True)
        self.tanh1_1 = torch.nn.Tanh()
        self.tanh1_2 = torch.nn.Tanh()
        self.tanh3_1 = torch.nn.Tanh()
    def forward(self, x0):
        x1 = self.tanh1_1(self.conv4_1(x0))
        x2 = self.tanh1_2(self.conv4_2(x1))
        x3 = self.conv7_1(x2)
        x4 = self.tanh3_1(self.conv8_1(x3))
        return x4
# Inputs to the model
x0 = torch.randn(1, 256, 224, 224)
