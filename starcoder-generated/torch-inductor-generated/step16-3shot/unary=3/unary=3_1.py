
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(32, 32, 6, stride=2, padding=0)
        self.conv2 = torch.nn.Conv1d(32, 16, 1, stride=1, padding=15)
        self.conv3 = torch.nn.Conv1d(496, 16, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv1d(16, 128, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv1d(128, 32, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv1d(32, 64, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv1d(64, 128, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv1d(128, 64, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv1d(64, 16, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv1d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
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
        v13 = torch.nn.functional.pad(v12, (1, 1))
        v14 = self.conv3(v13)
        v15 = v14 * 0.5
        v16 = v14 * 0.7071067811865476
        v17 = torch.erf(v16)
        v18 = v17 + 1
        v19 = v15 * v18
        v20 = torch.nn.functional.pad(v19, (1, 1))
        v21 = self.conv4(v20)
        v22 = v21 * 0.5
        v23 = v21 * 0.7071067811865476
        v24 = torch.erf(v23)
        v25 = v24 + 1
        v26 = v22 * v25
        v27 = torch.nn.functional.pad(v26, (3, 3))
        v28 = self.conv5(v27)
        v29 = v28 * 0.5
        v30 = v28 * 0.7071067811865476
        v31 = torch.erf(v30)
        v32 = v31 + 1
        v33 = v29 * v32
        v34 = torch.nn.functional.pad(v33, (1, 1))
        v35 = self.conv6(v34)
        v36 = v35 * 0.5
        v37 = v35 * 0.7071067811865476
        v38 = torch.erf(v37)
        v39 = v38 + 1
        v40 = v36 * v39
        v41 = torch.nn.functional.pad(v40, (1, 1))
        v42 = self.conv7(v41)
        v43 = v42 * 0.5
        v44 = v42 * 0.7071067811865476
        v45 = torch.erf(v44)
        v46 = v45 + 1
        v47 = v43 * v46
        v48 = torch.nn.functional.pad(v47, (0, 0))
        v49 = self.conv8(v48)
        v50 = v49 * 0.5
        v51 = v49 * 0.7071067811865476
        v52 = torch.erf(v51)
        v53 = v52 + 1
        v54 = v50 * v53
        v55 = torch.nn.functional.pad(v54, (0, 0))
        v56 = self.conv9(v55)
        v57 = v56 * 0.5
        v58 = v56 * 0.7071067811865476
        v59 = torch.erf(v58)
        v60 = v59 + 1
        v61 = v57 * v60
        v62 = torch.nn.functional.pad(v61, (1, 1))
        v63 = self.conv10(v62)
        return v63
# Inputs to the model
x1 = torch.randn(1, 32, 629)
