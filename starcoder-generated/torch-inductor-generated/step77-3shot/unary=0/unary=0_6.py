
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(19, 30, 2, stride=1, padding=0)
        self.conv_1 = torch.nn.BatchNorm2d(30)
        self.conv_2 = torch.nn.ReLU()
        self.conv_3 = torch.nn.Conv2d(30, 30, 2, stride=1, padding=0)
        self.conv_5 = torch.nn.BatchNorm2d(30)
        self.conv_6 = torch.nn.ReLU()
        self.conv_7 = torch.nn.Conv2d(30, 30, 1, stride=1, padding=0)
        self.conv_9 = torch.nn.BatchNorm2d(30)
        self.conv_10 = torch.nn.ReLU()
        self.conv_11 = torch.nn.Conv2d(30, 30, 2, stride=1, padding=0)
        self.conv_13 = torch.nn.BatchNorm2d(30)
        self.conv_14 = torch.nn.ReLU()
        self.conv_15 = torch.nn.Conv2d(30, 30, 1, stride=1, padding=0)
        self.conv_17 = torch.nn.BatchNorm2d(30)
        self.conv_18 = torch.nn.ReLU()
        self.conv_19 = torch.nn.Conv2d(30, 30, 2, stride=1, padding=0)
        self.conv_21 = torch.nn.BatchNorm2d(30)
        self.conv_22 = torch.nn.ReLU()
        self.conv_23 = torch.nn.Conv2d(30, 30, 1, stride=1, padding=0)
        self.conv_25 = torch.nn.BatchNorm2d(30)
        self.conv_26 = torch.nn.ReLU()
        self.conv_27 = torch.nn.Conv2d(30, 30, 2, stride=1, padding=0)
        self.conv_29 = torch.nn.BatchNorm2d(30)
        self.conv_30 = torch.nn.ReLU()
        self.conv_31 = torch.nn.Conv2d(30, 30, 1, stride=1, padding=0)
        self.conv_33 = torch.nn.BatchNorm2d(30)
        self.conv_34 = torch.nn.ReLU()
        self.conv_35 = torch.nn.Conv2d(30, 1, 1, stride=1, padding=0)
        self.conv_37 = torch.nn.BatchNorm2d(1)
        self.conv_38 = torch.nn.ReLU()
    def forward(self, x138):
        v1 = self.conv_0(x138)
        v2 = self.conv_1(v1)
        v3 = self.conv_2(v2)
        v4 = self.conv_3(v3)
        v5 = self.conv_5(v4)
        v6 = self.conv_6(v5)
        v7 = self.conv_7(v6)
        v8 = self.conv_9(v7)
        v9 = self.conv_10(v8)
        v10 = self.conv_11(v9)
        v11 = self.conv_13(v10)
        v12 = self.conv_14(v11)
        v13 = self.conv_15(v12)
        v14 = self.conv_17(v13)
        v15 = self.conv_18(v14)
        v16 = self.conv_19(v15)
        v17 = self.conv_21(v16)
        v18 = self.conv_22(v17)
        v19 = self.conv_23(v18)
        v20 = self.conv_25(v19)
        v21 = self.conv_26(v20)
        v22 = self.conv_27(v21)
        v23 = self.conv_29(v22)
        v24 = self.conv_30(v23)
        v25 = self.conv_31(v24)
        v26 = self.conv_33(v25)
        v27 = self.conv_34(v26)
        v28 = self.conv_35(v27)
        v29 = self.conv_37(v28)
        v30 = self.conv_38(v29)
        v31 = v30 * 0.5
        v32 = v30 * v30
        v33 = v32 * v30
        v34 = v33 * 0.044715
        v35 = v30 + v34
        v36 = v35 * 0.7978845608028654
        v37 = torch.tanh(v36)
        v38 = v37 + 1
        v39 = v31 * v38
        return v39
# Inputs to the model
x138 = torch.randn(1, 19, 93, 51)
