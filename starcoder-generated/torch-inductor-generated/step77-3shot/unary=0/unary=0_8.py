
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(7, 10, 2, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(10, 8, 3, stride=2, padding=0)
        self.conv_3 = torch.nn.Conv2d(8, 5, 1, stride=1, padding=0)
        self.conv_4 = torch.nn.Conv2d(5, 4, 1, stride=1, padding=0)
        self.conv_5 = torch.nn.Conv2d(4, 2, 1, stride=1, padding=0)
        self.conv_6 = torch.nn.Conv2d(2, 9, 2, stride=1, padding=0)
    def forward(self, x2):
        v1 = self.conv_1(x2)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv_2(v10)
        v12 = v11 * 0.5
        v13 = v11 * v11
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v12 * v19
        v21 = self.conv_3(v20)
        v22 = v21 * 0.5
        v23 = v21 * v21
        v24 = v23 * v21
        v25 = v24 * 0.044715
        v26 = v21 + v25
        v27 = v26 * 0.7978845608028654
        v28 = torch.tanh(v27)
        v29 = v28 + 1
        v30 = v22 * v29
        v31 = self.conv_4(v30)
        v32 = v31 * 0.5
        v33 = v31 * v31
        v34 = v33 * v31
        v35 = v34 * 0.044715
        v36 = v31 + v35
        v37 = v36 * 0.7978845608028654
        v38 = torch.tanh(v37)
        v39 = v38 + 1
        v40 = v32 * v39
        v41 = self.conv_5(v40)
        v42 = v41 * 0.5
        v43 = v41 * v41
        v44 = v43 * v41
        v45 = v44 * 0.044715
        v46 = v41 + v45
        v47 = v46 * 0.7978845608028654
        v48 = torch.tanh(v47)
        v49 = v48 + 1
        v50 = v42 * v49
        v51 = self.conv_6(v50)
        v52 = v51 * 0.5
        v53 = v51 * v51
        v54 = v53 * v51
        v55 = v54 * 0.044715
        v56 = v51 + v55
        v57 = v56 * 0.7978845608028654
        v58 = torch.tanh(v57)
        v59 = v58 + 1
        v60 = v52 * v59
        return v20 + v30 + v40 + v50 + v60
# Inputs to the model
x2 = torch.randn(3, 7, 5, 8)
