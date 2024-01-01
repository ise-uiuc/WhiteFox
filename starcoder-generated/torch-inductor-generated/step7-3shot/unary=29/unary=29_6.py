
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0018913522922686085, max_value=0.004538534175171347):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 3, 3, stride=1, padding=0)
        self.maxpool = torch.nn.MaxPool2d(3, 2, padding=0)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
        self.padding = torch.nn.ReflectionPad2d(0)
        self.conv2_pad = torch.nn.Conv2d(224, 256, 3, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.maxpool2 = torch.nn.MaxPool2d(3, 2, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2_1 = torch.nn.Conv2d(112, 256, 1, stride=1, padding=0, groups=1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.upsample = torch.nn.Upsample(scale_factor=1.0, mode='nearest')
        self.conv1 = torch.nn.Conv2d(112, 256, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv1_1 = torch.nn.Conv2d(112, 256, 1, stride=1, padding=0, groups=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v3 = self.conv2(v1)
        v4 = self.maxpool(v3)
        v5 = self.conv_transpose1(v4)
        v6 = self.padding(v5)
        v7 = self.conv2_pad(v6)
        v8 = self.relu2(v7)
        v9 = self.maxpool2(v8)
        v10 = self.relu(v9)
        v11 = self.conv2_1(v10)
        v12 = self.relu1(v11)
        v13 = self.relu6(v12)
        v14 = self.upsample(v13)
        v15 = self.conv1(v14)
        v16 = self.conv1_1(v15)
        v17 = self.relu(v16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 3, 320, 320)
