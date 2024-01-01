
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(39, 72, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = torch.nn.ReLU() # Replace
        self.conv3d = torch.nn.Conv3d(43, 254, kernel_size=(1,1,1), stride=(2,1,1), padding=(1,0,0), bias=False)
        self.relu2 = torch.nn.ReLU() # Replace
        self.conv1d = torch.nn.Conv1d(20, 60, kernel_size=1, stride=1, padding=0, bias=False)
        self.transpose3d = torch.nn.ConvTranspose3d(813, 851, kernel_size=(1, 7, 3), stride=(1, 2, 2), padding=(0, 5, 3), output_padding=(0, 0, 0), groups=4, bias=False, dilation=2)
        self.relu3 = torch.nn.ReLU() # Replace
        self.conv3d2 = torch.nn.Conv3d(1911, 332, kernel_size=1, stride=1, padding=0, bias=False)
        # self.transpose3d = torch.nn.ConvTranspose3d() # Replace
        # self.relu3 = torch.nn.ReLU() # Replace
        # self.conv3d2 = torch.nn.Conv3d() # Replace
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv2d(x1)
        v2 = self.relu(v1)
        v3 = v2 * x6
        v4 = self.conv3d(v3)
        v5 = self.relu2(v4)
        v6 = self.conv1d(v5)
        v7 = self.transpose3d(v6)
        v8 = self.relu3(v7)
        v9 = self.conv3d2(v8)
        vR = v9 * x5
        return vR
# Inputs to the model
x1 = torch.randn(10, 39, 92, 92)
x2 = torch.randn(10, 43, 120, 120, 44)
x3 = torch.randn(10, 20, 451)
x4 = torch.randn(10, 813, 36, 52, 52)
x5 = torch.randn(10, 1911, 44, 44, 44)
x6 = torch.randn(10)
