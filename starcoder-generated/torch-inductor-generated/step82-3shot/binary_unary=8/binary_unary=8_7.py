
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv2 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv3 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv4 = torch.nn.Conv2d(3, 128, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv5 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv6 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv7 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv8 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv9 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv10 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1, x2):
        v1 = self.pointwise_conv1(x1)
        v2 = self.pointwise_conv2(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.pointwise_conv3(x1)
        v6 = self.pointwise_conv4(x1)
        v7 = v5 + v6
        v8 = torch.relu(v7)
        v9 = self.pointwise_conv5(x2)
        v10 = self.pointwise_conv6(x2)
        v11 = v9 + v10
        v12 = torch.relu(v11)
        v13 = self.pointwise_conv7(x2)
        v14 = self.pointwise_conv7(x2)
        v15 = self.pointwise_conv8(x2)
        v16 = self.pointwise_conv9(x2)
        v17 = v13 + v14 + v15 + v16
        v18 = torch.relu(v17)
        v19 = self.pointwise_conv10(x2)
        v20 = self.pointwise_conv10(x2)
        v21 = self.pointwise_conv10(x2)
        v22 = self.pointwise_conv10(x2)
        v23 = v19 + v20 + v21 + v22
        v24 = torch.relu(v23)
        v25 = v18 + v24
        v26 = torch.relu(v25)
        v27 = v4 + v8 + v26
        return v27
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
