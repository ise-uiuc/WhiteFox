
class Block2(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channel_size, channel_size, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channel_size)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channel_size, channel_size, 3, 1, 1, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = self.act1(self.bn1(self.conv1(concatenated_tensor)))
        v3 = torch.nn.functional.interpolate(v2, size=(44, 44), align_corners=False)
        v4 = (v2, v3)
        return v4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v0 = torch.nn.Parameter(torch.ones((1, 3, 224, 224), dtype=torch.float32), requires_grad=True)
        self.features = [Block2(32)]
        self.features = torch.nn.Sequential(*self.features)
        self.features_1 = [torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(64)]
        self.features_2 = [torch.nn.ReLU()]
        self.features_3 = [torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(64)]
        self.features_4 = [torch.nn.ReLU()]
        self.features_5 = [Block1(64)]
        self.features_6 = [torch.nn.Conv2d(64, 64, 33, 1, 11, bias=False), torch.nn.BatchNorm2d(64)]
        self.features_7 = [torch.nn.ReLU()]
        self.features_8 = [Block1(32)]
        self.features_9 = [torch.nn.Conv2d(64, 64, 17, 1, 11, bias=False), torch.nn.BatchNorm2d(64)]
        self.features_10 = [torch.nn.ReLU()]
        self.features_11 = [torch.nn.Conv2d(64, 64, 9, 1, 5, bias=False), torch.nn.BatchNorm2d(64), torch.nn.AvgPool2d(7, 1, 0)]
        self.features_12 = [torch.nn.ReLU()]
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = self.features(concatenated_tensor)
        v6 = self.features_1[0](v2)
        v7 = self.features_1[1](v6)
        v3 = self.features_2[0](v7)
        v8 = self.features_3[0](v3)
        v9 = self.features_3[1](v8)
        v4 = self.features_4[0](v9)
        v10 = self.features_5[0](v4)
        v11 = self.features_5[1](v10)
        v5 = self.features_6[0](v11)
        v12 = self.features_6[1](v5)
        v13 = torch.cat([v1, v12], dim=1)
        v14 = self.features_7[0](v13)
        v15 = self.features_8[0](v14)
        v16 = self.features_9[0](v15)
        v17 = self.features_9[1](v16)
        v18 = self.features_10[0](v17)
        v19 = self.features_11[0](v18)
        v100 = self.features_12[0](v19)
        return v100
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
