
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv45 = torch.nn.Conv2d(45, 1, 1, stride=1, padding=0)
        self.conv23 = torch.nn.Conv2d(45, 11, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(11, 7, 1, stride=1, padding=0)
        self.conv97 = torch.nn.Conv2d(22, 4, 3, stride=1, padding=1)
        self.conv65 = torch.nn.Conv2d(4, 23, 3, stride=1, padding=1)
        self.conv87 = torch.nn.Conv2d(49, 1, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(49, 54, 7, stride=3, padding=0)
        self.conv89 = torch.nn.Conv2d(54, 97, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(97, 153, 3, stride=1, padding=1)
        self.conv78 = torch.nn.ConvTranspose2d(153, 97, 1, stride=1, padding=0)
        self.conv91 = torch.nn.ConvTranspose2d(97, 32, 3, stride=1, padding=1)
        self.conv46 = torch.nn.ConvTranspose2d(32, 31, 3, stride=1, padding=1)
        self.conv12 = torch.nn.ConvTranspose2d(31, 32, 7, stride=3, padding=0)
        self.conv55 = torch.nn.Conv2d(74, 86, 1, stride=1, padding=0)
        self.conv36 = torch.nn.ConvTranspose2d(86, 83, 3, stride=1, padding=1)
        self.conv58 = torch.nn.ConvTranspose2d(83, 116, 3, stride=1, padding=1)
        self.conv47 = torch.nn.ConvTranspose2d(116, 43, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv45(x1)
        v2 = self.conv23(x1)
        v3 = self.conv6(v2)
        v4 = self.conv97(v3)
        v5 = self.conv65(v4)
        v6 = self.conv87(v5)
        v7 = v6 + v1
        v8 = self.conv5(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.conv89(v13)
        v15 = self.conv4(v14)
        v16 = v15 * 0.5
        v17 = v15 * 0.7071067811865476
        v18 = torch.erf(v17)
        v19 = v18 + 1
        v20 = v16 * v19
        v21 = self.conv78(v20)
        v22 = self.conv91(v21)
        v23 = self.conv46(v22)
        v24 = self.conv12(v23)
        v25 = v24 * 0.5
        v26 = v24 * 0.7071067811865476
        v27 = torch.erf(v26)
        v28 = v27 + 1
        v29 = v25 * v28
        v30 = self.conv55(v29)
        v31 = self.conv36(v30)
        v32 = self.conv58(v31)
        v33 = self.conv47(v32)
        v34 = v33 * 0.5
        v35 = v33 * 0.7071067811865476
        v36 = torch.erf(v35)
        v37 = v36 + 1
        v38 = v34 * v37
        return v38
# Inputs to the model
x1 = torch.randn(1, 45, 23, 27)
