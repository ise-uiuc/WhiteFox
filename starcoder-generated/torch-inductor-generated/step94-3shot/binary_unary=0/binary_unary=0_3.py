
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5):
        t1 = self.conv1(x1)
        t2 = t1 + x2
        t3 = torch.softmax(t2)
        t4 = t3
        t5 = self.conv2(t4)
        t6 = t1 + t5
        t7 = torch.softmax(t6)
        t8 = t7
        t6 = self.conv3(t8)
        t13 = t1 + t6
        t14 = torch.softmax(t13)
        t15 = self.conv4(t14)
        t16 = t1 + t15
        t17 = torch.softmax(t16)
        t18 = t17
        t13 = self.conv6(t18)
        t20 = t1 + t13
        t21 = torch.softmax(t20)
        t22 = t21
        t13 = self.conv5(t7)
        t23 = t1 + t13
        t24 = torch.softmax(t23)
        t25 = t24
        t27 = t1 + t22
        t28 = torch.softmax(t27)
        t29 = t28
        t22 = self.conv5(t14)
        t30 = t1 + t22
        t31 = torch.softmax(t30)
        t22 = self.conv6(t17)
        t32 = t1 + t22
        t33 = torch.softmax(t32)
        t17 = self.conv2(t14)
        t34 = t1 + t17
        t35 = torch.softmax(t34)
        t14 = self.conv3(t17)
        t14 = self.conv3(t30)
        t36 = t1 + t14
        t37 = torch.softmax(t36)
        t17 = self.conv6(t37)
        t38 = t1 + t17
        t39 = torch.softmax(t38)
        v1 = t1 + t32 + t39
        v2 = torch.sigmoid(v1)
        v3 = t1 + t24 + t35
        v4 = torch.sigmoid(v3)
        v5 = v4
        v6 = self.conv5(t14)
        v6 = v1 + v6
        v7 = torch.sigmoid(v6)
        v8 = v7
        v6 = self.conv5(t7)
        v13 = t1 + v6
        v14 = torch.sigmoid(v13)
        v15 = v14
        v13 = t1 + v16 + v3
        v19 = t1 + v15 + v3
        v14 = torch.sigmoid(v19)
        v15 = v14
        v19 = self.conv5(t30)
        v0 = v1 + v19
        v9 = torch.sigmoid(v0)
        v19 = self.conv6(t38)
        v20 = v1 + v19
        v21 = torch.sigmoid(v20)
        v19 = self.conv5(t37)
        v22 = v1 + v19
        v23 = torch.sigmoid(v22)
        v19 = t1 + v15 + v3 + v21 + v23 + v23 + v15 + v3 + v15
        v27 = t1 + v15 + v3 + v23 + v3 + v3 + v15 + v15 + v15
        v28 = torch.sigmoid(v27)
        v29 = v28
        v19 = self.conv5(t6)
        v31 = v1 + v19
        v32 = torch.sigmoid(v31)
        v19 = self.conv5(v6)
        v33 = v1 + v19
        v34 = torch.sigmoid(v33)
        v19 = t1 + v8 + v23 + v15 + v21 + v15 + v8 + v15 + v23
        v38 = v1 + v14 + v8 + v34 + v15 + v15 + v23 + v8
        v39 = torch.sigmoid(v38)
        v40 = v39
        v36 = v1 + v23 + v34 + v17 + v33 + v8 + v15 + v13 + v15
        v36 = torch.sigmoid(v36)
        v37 = v36
        v36 = t1 + v7 + v21 + v32 + v15 + v23 + v21 + v8 + v14
        v43 = t1 + v40 + v40 + v40 + v40 + v40 + v40 + v40 + v40
        v44 = torch.sigmoid(v43)
        v45 = v44
        x1 = t1 + v8 + v45 + t6 + v15 + v15 + x1
        x2 = t1 + v8 + v21 + v5 + v15 + v15 + x2
        x3 = t1 + v0 + v3 + v32 + x3
        x4 = t1 + v9 + v3 + v5 + v15 + v15 + x4
        x5 = t1 + v23 + self.conv5(x5)
        x6 = t1 + v1 + x5
        x6 = t1 + x5
        x5 = torch.sigmoid(x5)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
x5 = torch.randn(1, 3, 64, 64)
