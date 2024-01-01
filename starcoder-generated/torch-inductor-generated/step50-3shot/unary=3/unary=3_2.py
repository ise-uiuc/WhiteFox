
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv6 = torch.nn.Conv2d(85, 53, 7, stride=1, padding=3)
        self.conv8 = torch.nn.Conv2d(53, 111, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(111, 50, 1, stride=1, padding=0)
        self.conv12 = torch.nn.Conv2d(50, 142, 7, stride=1, padding=3)
        self.conv14 = torch.nn.Conv2d(142, 70, 1, stride=1, padding=0)
        self.conv = torch.nn.ConvTranspose2d(70, 1, 2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv6(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv8(v6)
        v8 = v7 * 0.5
        v9 = torch.tanh(v8)
        v10 = v9 * 0.5
        v11 = v9 * 0.7071067811865476
        v12 = torch.erf(v11)
        v13 = v12 + 1
        v14 = v10 * v13
        v15 = self.conv10(v14)
        v16 = v15 * 0.5
        v17 = torch.tanh(v16)
        v18 = v17 * 0.5
        v19 = v17 * 0.7071067811865476
        v20 = torch.erf(v19)
        v21 = v20 + 1
        v22 = v18 * v21
        v23 = self.conv12(v22)
        v24 = v23 * 0.5
        v25 = torch.tanh(v24)
        v26 = v25 * 0.5
        v27 = v25 * 0.7071067811865476
        v28 = torch.erf(v27)
        v29 = v28 + 1
        v30 = v26 * v29
        v31 = self.conv14(v30)
        v32 = v31 * 0.5
        v33 = torch.tanh(v32)
        v34 = v33 * 0.5
        v35 = v33 * 0.7071067811865476
        v36 = torch.erf(v35)
        v37 = v36 + 1
        v38 = v34 * v37
        v39 = v38.to(device="cpu").detach().numpy()
        #v40 = v39 + torch.randn(v38.shape)
        v40 = np.array(v39,dtype='float32')
        v41 = torch.from_numpy(v40)
        v42 = v41 + torch.zeros_like(v38)
        v43 = self.conv(v42)
        return v43
# Inputs to the model
x1 = torch.randn(1, 85, 31, 33)
