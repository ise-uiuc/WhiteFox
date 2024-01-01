
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv3 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv4 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv5 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv6 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv7 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv8 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv9 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv10 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv11 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0, bias=False)
        self.conv12 = torch.nn.Conv2d(63, 64, 1, stride=1, padding=0, bias=False)
        self.conv13 = torch.nn.Conv2d(63, 64, 1, stride=1, padding=0, bias=False)
        self.conv14 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False)
        self.pad15 = torch.nn.ConstantPad2d((1, 1, 1, 1), 0.0)
        self.conv16 = torch.nn.Conv2d(63, 16, 3, stride=1, padding=0, bias=True)
        self.relu17 = torch.nn.ReLU(inplace=True)
        self.conv18 = torch.nn.Conv2d(63, 16, 3, stride=1, padding=1, bias=True)
        self.relu19 = torch.nn.ReLU(inplace=True)
    def forward(self, x1):    
        v1 = self.conv1(x1)
        v2 = v1 - 0.01
        v3 = F.relu(v2)
        v4 = self.conv2(v1)
        v5 = torch.cat([v3, v4], 1)
        v6 = torch.abs(v5)
        v7 = v6 + 1.08
        v8 = F.relu(v7)
        v9 = self.conv3(v5)
        v10 = v9 - 0.02
        v11 = F.relu(v10)
        v12 = torch.abs(v11)
        v13 = v12 * 3.09
        v14 = v13 - 1.08
        v15 = F.relu(v14)
        v16 = torch.floor(v15)
        v17 = v16 - 1.05
        v18 = F.relu(v17)
        v19 = self.conv4(v11)
        v20 = self.conv5(self.conv6(self.conv7(self.conv8(self.conv9(self.conv10(self.conv11(v19)))))))
        v21 = v20 + 0.03
        v22 = F.relu(v21)
        v23 = v22 + 1.08
        v24 = F.relu(v23)
        v25 = self.conv12(v11)
        v26 = torch.cat([v25, v24], 1)
        v27 = v26 + 1.07
        v28 = F.relu(v27)
        v29 = self.conv13(self.conv14(v28))
        v30 = self.pad15(v29)
        v31 = v30 / 0.03
        v32 = torch.floor(v31)
        x2 = self.conv16(v30)
        x3 = v32 - 2.05
        x4 = self.relu17(x2)
        x5 = self.conv18(x3)
        x6 = x4 + x5
        x7 = self.relu19(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 64, 37, 37)
