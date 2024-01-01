
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(512, 512, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 4.45
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 99.88
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 11.99
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 14.9
        v12 = F.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 - 29.9
        v15 = F.relu(v14)
        v16 = self.conv6(v15)
        v17 = v16 - 39.9
        v18 = F.relu(v17)
        v19 = self.conv7(v18)
        v20 = v19 - 745.614
        v21 = F.relu(v20)
        t1 = torch._convolution(x1, v21, None, [3, 3], [1, 1], [1, 1])
        t2 = self.relu(t1)
        t3 = self.conv1(t2)
        t4 = t3 - -0.6026
        t5 = self.relu(t4)
        t6 = self.conv2(t5)
        t7 = t6 - -3.965
        t8 = self.relu(t7)
        t9 = self.conv3(t8)
        t10 = t9 - -4.2331
        t11 = self.relu(t10)
        t12 = self.conv4(t11)
        t13 = t12 - -2.6922
        t14 = self.relu(t13)
        t15 = self.conv5(t14)
        t16 = t15 - 5.0648
        t17 = self.relu(t16)
        t18 = self.conv6(t17)
        t19 = t18 - 1.6424
        t20 = self.relu(t19)
        t21 = self.conv7(t20)
        t22 = t21 - -3.413
        t23 = self.relu(t22)
        t24 = self.conv8(t23)
        t25 = t24 - 0.0574
        t26 = self.relu(t25)
        t27 = self.conv9(t26)
        t28 = t27 - 4.1603
        t29 = self.relu(t28)
        t30 = self.conv10(t29)
        t31 = t30 - 5.2759
        t32 = self.relu(t31)
        t33 = self.conv11(t32)
        t34 = t33 - 85.1385
        t35 = self.relu(t34)
        return t35, v2
# Inputs to the model
x1 = torch.randn(1, 1, 192, 192)
