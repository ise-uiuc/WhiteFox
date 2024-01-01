
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1) + v1
        v3 = self.conv(v1)
        v4 = torch.relu(v2) + v2
        v5 = self.conv(v2)
        v6 = torch.relu(v3) + v3
        v7 = self.conv(v4)
        v8 = torch.relu(v5) + v5
        v9 = self.conv(v6)
        v10 = torch.relu(v8) + v8
        v11 = self.conv(v8)
        v12 = torch.relu(v9) + v9
        v13 = self.conv(v10)
        v14 = torch.relu(v11) + v11
        v15 = self.conv(v12)
        v16 = torch.relu(v13) + v13
        v17 = self.conv(v14)
        v18 = torch.relu(v15) + v15
        v19 = self.conv(v16)
        v20 = torch.relu(v17) + v17
        v21 = self.conv(v18)
        v22 = torch.relu(v19) + v19
        v23 = self.conv(v20)
        v24 = torch.relu(v21) + v21
        v25 = self.conv(v22)
        v26 = torch.relu(v23) + v23
        v27 = self.conv(v24)
        v28 = torch.relu(v25) + v25
        v29 = self.conv(v26)
        v30 = torch.relu(v27) + v27
        v31 = self.conv(v28)
        v32 = torch.relu(v29) + v29
        v33 = self.conv(v30)
        v34 = torch.relu(v31) + v31
        v35 = self.conv(v32)
        v36 = torch.relu(v33) + v33
        v37 = self.conv(v34)
        v38 = torch.relu(v35) + v35
        v39 = self.conv(v36)
        v40 = torch.relu(v37) + v37
        v41 = self.conv(v38)
        v42 = torch.relu(v39) + v39
        v43 = self.conv(v40)
        v44 = torch.relu(v41) + v41
        v45 = self.conv(v42)
        v46 = torch.relu(v43) + v43
        v47 = self.conv(v44)
        v48 = torch.relu(v45) + v45
        return v48
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
