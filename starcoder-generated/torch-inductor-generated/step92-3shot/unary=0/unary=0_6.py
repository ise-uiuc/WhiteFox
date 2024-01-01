
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 30, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(30, 40, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(40, 50, 10, stride=11, padding=10)
        self.conv4 = torch.nn.Conv2d(50, 60, 2, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(60, 70, 10, stride=1, padding=2)
        self.conv6 = torch.nn.Conv2d(70, 80, 3, stride=2, padding=3)
        self.conv7 = torch.nn.Conv2d(80, 90, 3, stride=3, padding=1)
        self.conv8 = torch.nn.Conv2d(90, 70, 6, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(70, 2, 20, stride=2, padding=2)
        self.conv10 = torch.nn.Conv2d(2, 24, 6, stride=10, padding=4)
    def forward(self, x24):
        v1 = self.conv1(x24)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv8(v10)
        v12 = torch.tanh(x24)
        v13 = self.conv9(v8)
        v14 = v12 * v13
        v15 = v1 + v14
        v16 = self.conv2(v15)
        v17 = (v11 + v15) * v11
        v18 = self.conv3(v17)
        v19 = v18 * v16
        v20 = v11 + v19
        v21 = v19 + v20
        v22 = self.conv4(v21)
        v23 = (v22 + v23) * v11
        v24 = v11 * v18
        v25 = torch.tanh(v11)
        v26 = self.conv5(v21)
        v27 = torch.tanh(v25)
        v28 = self.conv6(v27)
        v29 = torch.tanh((v27 + v28) * v11)
        v30 = self.conv7(v29)
        v31 = torch.tanh(v24)
        v32 = self.conv7(v19)
        v33 = v31 * v27
        v34 = self.conv6(v19)
        v35 = v11 + v25
        v36 = v34 + v35
        v37 = self.conv5(v36)
        v38 = self.conv5(v1)
        v39 = v1 + v38
        v40 = v38 + v25
        v41 = v33 * v24
        v42 = self.conv6(v20)
        v43 = v1 + v38
        v44 = v39 * v40
        v45 = self.conv6(v23)
        v46 = v1 + v39
        v47 = v44 + v43
        v48 = v35 + v46
        v49 = v25 + v39
        v50 = self.conv8(v48)
        v51 = self.conv10(v48)
        v52 = v11 + 0.01
        v53 = self.conv6(v17)
        v54 = self.conv5(v35)
        v55 = self.conv9(v31)
        v56 = self.conv5(v26)
        v57 = v1 + v31
        v58 = self.conv5(v38)
        v59 = v54 * v1
        v60 = v11 * v58
        v61 = v58 + v15
        v62 = self.conv6(v61)
        v63 = self.conv1(v60)
        v64 = v49 + v52
        v65 = v56 * v1
        v66 = v1 + v28
        v67 = self.conv10(v37)
        v68 = v27 + 0.10000000149011612
        v69 = v51 + v63
        v70 = v59 + v15
        v71 = v1 + v66
        v72 = self.conv7(v19)
        v73 = v39 * v27
        v74 = torch.softmax(v61, dim=1)
        return ((v60 + v70 + v71) * (v55 + v57) + (v67 + 0.0039999999105930335) * (v22 + v29) + v11 + (v74 + v62 + v32 + v72) * v15 * (v19 + v73) * v69 + 0.5 * v1 * v64 * v50 * v68).permute(2, 3, 0, 1).flatten(1) # Add.permute(2, 3, 0, 1) to the last line in the forward method (the last line in the backward method)
# Inputs to the model
x24 = torch.randn(1, 3, 4, 9)
