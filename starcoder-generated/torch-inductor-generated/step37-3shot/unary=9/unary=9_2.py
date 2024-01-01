
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv1 = torch.nn.Conv2d(8, 16, 1)
        self.other_conv2 = torch.nn.Conv2d(16, 32, 1)
        self.other_conv3 = torch.nn.Conv2d(32, 16, 1)
        self.other_conv4 = torch.nn.Conv2d(16, 8, 1)
        self.other_conv5 = torch.nn.Conv2d(8, 8, 1)
        self.other_conv6 = torch.nn.Conv2d(8, 8, 1)
        self.other_conv7 = torch.nn.Conv2d(8, 8, 1)
    def forward(self, x1):
        v1 = 3 + self.conv(x1)  # t1
        v2 = v1 - 1  # t2
        v3 = v1 * 1  # t3
        v4 = v1 / 1  # t4
        v5 = 1 - v4  # t5
        v6 = 1 + v2  # t6
        v7 = 1 * v3  # t7
        v8 = v7 / 1  # t8
        v9 = v1.neg()  # t9
        v10 = v9  # t10
        v11 = v8 * 6  # t11
        v12 = v6 * 6  # t12
        v13 = v12 - 6  # t13
        v14 = v13 / 6  # t14
        v15 = 1 / v5  # t15
        v16 = v14 * v15  # t16
        v17 = v11 / 6  # t17
        v18 = 3 + v17  # t18
        v19 = 6 / v8  # t19
        v20 = 6 / v6  # t20
        v21 = torch.clamp_max(v20, 6)  # t21
        v22 = torch.clamp_min(v19, 0)  # t22
        v23 = v21 * v22  # t23
        v24 = v18 * 6  # t24
        v25 = v24 / 6  # t25
        v26 = v23 * 6  # t26
        v27 = v25 * v26  # t27
        v28 = v16 * v27  # t28
        v29 = v3 + 3  # t29
        v30 = 6 / v10  # t30
        v31 = torch.clamp_min(v30, 0)  # t31
        v32 = v31 * 6  # t32
        v33 = v29 * v32  # t33
        x2 = v28 + v33  # t34
        v35 = 3 / x2  # t35
        v36 = v16.div(6)  # t36
        return self.other_conv7(self.other_conv6(self.other_conv5(self.other_conv4(self.other_conv3(self.other_conv2(self.other_conv1(v35)))))))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
