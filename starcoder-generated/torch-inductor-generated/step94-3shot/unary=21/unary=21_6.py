
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1024, 1024, kernel_size=[7, 7], stride=(1, 1), padding=(3, 3), bias=True)
        self.conv10 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv11 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv12 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv13 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv14 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv15 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv16 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv17 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv18 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv19 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv20 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv21 = torch.nn.Conv2d(1024, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv3 = torch.nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5 = torch.nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv6 = torch.nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv7 = torch.nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
        self.conv8 = torch.nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv9 = torch.nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=False)
    def forward(self, x0):
        v1 = self.conv1(x0)
        v2 = torch.tanh(v1)
        v3 = self.conv10(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv11(v4)
        v6 = torch.sigmoid(v5)
        v7 = v4 * v6
        v8 = self.conv12(v7)
        v9 = torch.tanh(v8)
        v10 = self.conv13(v9)
        v11 = torch.sigmoid(v10)
        v12 = v9 * v11
        v13 = v7 + v12
        v14 = self.conv14(v13)
        v15 = torch.tanh(v14)
        v16 = self.conv15(v15)
        v17 = torch.sigmoid(v16)
        v18 = v15 * v17
        v19 = v13 + v18
        v20 = self.conv16(v19)
        v21 = torch.tanh(v20)
        v22 = self.conv17(v21)
        v23 = torch.sigmoid(v22)
        v24 = v21 * v23
        v25 = v19 + v24
        v26 = self.conv18(v25)
        v27 = torch.tanh(v26)
        v28 = self.conv19(v27)
        v29 = torch.sigmoid(v28)
        v30 = v27 * v29
        v31 = v25 + v30
        v32 = self.conv2(v31)
        v33 = self.conv20(v32)
        v34 = self.conv21(v33)
        v35 = v32 + v34
        v36 = self.conv3(v35)
        v37 = torch.tanh(v36)
        v38 = self.conv4(v37)
        v39 = torch.sigmoid(v38)
        v40 = v37 * v39
        v41 = self.conv5(v40)
        v42 = torch.tanh(v41)
        v43 = self.conv6(v42)
        v44 = torch.sigmoid(v43)
        v45 = v42 * v44
        v46 = v40 + v45
        v47 = self.conv7(v46)
        v48 = torch.tanh(v47)
        v49 = self.conv8(v48)
        v50 = torch.sigmoid(v49)
        v51 = v48 * v50
        v52 = v46 + v51
        v53 = self.conv9(v52)
        v54 = torch.tanh(v53)
        return torch.min(v54, 2)[0]
# Inputs to the model
input_tensor = torch.randn(1, 1024, 16, 16)
