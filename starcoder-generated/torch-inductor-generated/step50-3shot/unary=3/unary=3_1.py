
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 14, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(14, 35, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(35, 16, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(16, 12, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(12, 11, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(11, 5, 5, stride=2, padding=0)
        self.conv7 = torch.nn.Conv2d(5, 11, 1, stride=1, padding=0)
        self.conv8 = torch.nn.ConvTranspose2d(11, 12, 3, stride=1, padding=0)
        self.conv9 = torch.nn.ConvTranspose2d(12, 16, 3, stride=2, padding=0)
        self.conv10 = torch.nn.ConvTranspose2d(16, 35, 1, stride=1, padding=0)
        self.conv11 = torch.nn.ConvTranspose2d(35, 14, 1, stride=1, padding=0)
        self.conv12 = torch.nn.ConvTranspose2d(14, 1, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.conv4(v13)
        v15 = self.conv5(v14)
        v16 = v15 * 0.5
        v17 = v15 * 0.7071067811865476
        v18 = torch.erf(v17)
        v19 = v18 + 1
        v20 = v16 * v19
        v21 = self.conv6(v20)
        v22 = self.conv7(v21)
        v23 = v22 * 0.5
        v24 = v22 * 0.7071067811865476
        v25 = torch.erf(v24)
        v26 = v25 + 1
        v27 = v23 * v26
        v28 = self.conv8(v27)
        v29 = self.conv9(v28)
        v30 = v29 * 0.5
        v31 = v29 * 0.7071067811865476
        v32 = torch.erf(v31)
        v33 = v32 + 1
        v34 = v30 * v33
        v35 = self.conv10(v34)
        v36 = self.conv11(v35)
        v37 = self.conv12(v36)
        return v37
# Inputs to the model
x1 = torch.randn(1, 1, 29, 29)
