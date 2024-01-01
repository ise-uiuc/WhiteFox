
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 3, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(6, 3, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(6, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v23 = v2 * v5
        v24 = self.conv2(v23)
        v25 = v24 * 0.5
        v26 = v24 * 0.7071067811865476
        v27 = torch.erf(v26)
        v28 = v27 + 1
        v29 = v25 * v28
        v30 = self.conv3(v29)
        v31 = v30 * 0.5
        v32 = v30 * 0.7071067811865476
        v33 = torch.erf(v32)
        v34 = v33 + 1
        v35 = v31 * v34
        v36 = self.conv4(v35)
        v37 = v36 * 0.5
        v38 = v36 * 0.7071067811865476
        v39 = torch.erf(v38)
        v40 = v39 + 1
        v41 = v37 * v40
        v42 = self.conv5(v41)
        v43 = v42 * 0.5
        v44 = v42 * 0.7071067811865476
        v45 = torch.erf(v44)
        v46 = v45 + 1
        v47 = v43 * v46
        v48 = self.conv6(v47)
        return v48
# Inputs to the model
x1 = torch.randn(1, 3, 77, 33)
