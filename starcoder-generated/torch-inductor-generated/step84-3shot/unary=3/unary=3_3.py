
class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        return v2
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v2 * 0.5
        return v3
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v2 * 0.7071067811865476
        return v3
class Model4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v2 * 0.7071067811865476
        v4 = torch.erf(v3)
        return v4
class Model5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v2 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        return v5
class Model6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v2 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v5 + 1
        return v6
class Model7(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        return v5
class Model8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        return v4
class Model9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1
        return v2
class Model10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1
        v3 = v2 + 1
        return v3
class Model11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
class Model12(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        return v12
class Model13(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v6 * 0.5
        v8 = v6 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v7 * v10
        v12 = self.conv2(v11)
        return v12
class Model14(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v6 * 0.5
        v8 = v6 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v7 * v10
        v12 = self.conv2(v11)
        v13 = v12 * 0.5
        v14 = v12 * 0.7071067811865476
        v15 = torch.erf(v14)
        v16 = v15 + 1
        v17 = v13 * v16
        return v17
class Model15(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v6 * 0.5
        v8 = v6 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v7 * v10
        v12 = self.conv2(v11)
        v13 = v12 * 0.5
        v14 = v12 * 0.7071067811865476
        v15 = torch.erf(v14)
        return v15
class Model16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
class Model17(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        return v4
class Model18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, 9, stride=1, padding=4)
        self.conv2 = torch.nn.Conv2d(6, 8, 9, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1
        v3 = self.conv2(v2)
        return v3
class Model19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 13, stride=1, padding=6)
        self.conv2 = torch.nn.Conv2d(4, 8, 13, stride=1, padding=6)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
class Model20(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1
        v3 = v2 + 1
        v4 = v3 + 1
        v5 = v4 + 1
        v6 = v5 + 1
        v7 = v6 * 0.5
        v8 = v6 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v7 * v10
        return v11
# Inputs to the model
x1 = torch.randn(1, 1, 76, 64)
