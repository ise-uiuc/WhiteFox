
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.Conv2d(15, 30, kernel_size=5, stride=3, padding=2)
        self.conv2d = torch.nn.Conv2d(89, 19, kernel_size=2, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2d2 = torch.nn.Conv2d(15, 28, kernel_size=3, stride=1, padding=2)
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 3, 1)
        v2 = self.conv_transpose(x1)
        v3 = v2 * 0.144489
        v4 = torch.nn.functional.prelu(v3, 0.18387221414566)
        v5 = torch.nn.functional.relu(v4)
        v6 = torch.nn.functional.relu(v5)
        v7 = v6 + v1
        v8 = v7 * 0.264287
        v9 = torch.nn.functional.prelu(v8, 0.364788)
        v10 = torch.nn.functional.relu(v9)
        v11 = torch.nn.functional.relu(v10)
        v12 = v11.permute(0, 3, 1, 2)
        v13 = self.conv2d(v12)
        v14 = v13 * 0.591532
        v15 = torch.nn.functional.prelu(v14, 0.5161418518598849)
        v16 = torch.nn.functional.relu(v15)
        v17 = torch.nn.functional.relu(v16)
        v18 = v17 + v7
        v19 = v18 * 0.408153
        v20 = torch.nn.functional.prelu(v19, 0.3587806857304939)
        v21 = torch.nn.functional.relu(v20)
        v22 = torch.nn.functional.relu(v21)
        v23 = self.max_pool2d(v22)
        v24 = torch.nn.functional.relu(v23)
        v25 = v24.permute(0, 2, 3, 1)
        v26 = self.conv2d2(v25)
        v27 = v26 * 0.585397
        v28 = torch.nn.functional.prelu(v27, 0.4436416299021474)
        v29 = torch.nn.functional.relu(v28)
        v30 = torch.nn.functional.relu(v29)
        v31 = v30.permute(0, 3, 1, 2)
        return v31
# Inputs to the model
x1 = torch.randn(3, 15, 24, 24)
x2 = torch.randn(3, 15, 32, 256)
