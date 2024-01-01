
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 120, 1, stride=1, padding=0)
        self.max_pool2x2 = torch.nn.MaxPool2d((2, 2), stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(120, 100, 2, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(100, 50, 2, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(50, 15, 2, stride=1, padding=0)
        self.avg_pool2x2 = torch.nn.AvgPool2d((2, 2), stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(15, 46, 1, stride=1, padding=0)
        self.conv7 = torch.nn.ConvTranspose2d(46, 3, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(100, 46, 2, stride=1, padding=0)
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
        v13 = self.max_pool2x2(v12)
        v14 = self.conv3(v13)
        v15 = v14 * 0.5
        v16 = v14 * 0.7071067811865476
        v17 = torch.erf(v16)
        v18 = v17 + 1
        v19 = v15 * v18
        v20 = self.conv4(v19)
        v21 = v20 * 0.5
        v22 = v20 * 0.7071067811865476
        v23 = torch.erf(v22)
        v24 = v23 + 1
        v25 = v21 * v24
        v26 = self.conv5(v25)
        v27 = v26 * 0.5
        v28 = v26 * 0.7071067811865476
        v29 = torch.erf(v28)
        v30 = v29 + 1
        v31 = v27 * v30
        v32 = self.avg_pool2x2(v31)
        v33 = v32 * 0.5
        v34 = v32 * 0.7071067811865476
        v35 = torch.erf(v34)
        v36 = v35 + 1
        v37 = v33 * v36
        v38 = self.conv6(v37)
        v39 = v38 * 0.5
        v40 = v38 * 0.7071067811865476
        v41 = torch.erf(v40)
        v42 = v41 + 1
        v43 = v39 * v42
        v44 = torch.nn.functional.interpolate(v43, None, (12, 23), None)
        v45 = self.conv7(v44)
        v46 = torch.nn.functional.interpolate(v19, None, (12, 23), None)
        v47 = torch.nn.functional.interpolate(v13, None, (12, 23), None)
        v48 = torch.nn.functional.interpolate(v12, None, (191, 189), None)
        v49 = torch.nn.functional.relu(v47)
        v50 = torch.nn.functional.interpolate(v49, None, (22, 31), None)
        v51 = torch.nn.functional.interpolate(x1, None, (16, 31), None)
        v52 = torch.nn.functional.sigmoid(v50)
        v53 = torch.nn.functional.interpolate(v52, None, (23, 75), None)
        v54 = torch.nn.functional.relu(v51)
        v55 = torch.nn.functional.interpolate(v54, None, (45, 95), None)
        v56 = torch.nn.functional.relu(v55)
        v57 = torch.nn.functional.interpolate(v56, None, (54, 41), None)
        v58 = torch.nn.functional.interpolate(v48, None, (54, 41), None)
        v59 = torch.nn.functional.interpolate(v57, None, (185, 47), None)
        v60 = self.conv8(v59)
        v61 = v60 * 0.5
        v62 = v60 * 0.7071067811865476
        v63 = torch.erf(v62)
        v64 = v63 + 1
        v65 = v61 * v64
        return v65
# Inputs to the model
x1 = torch.randn(1, 3, 63, 49)
