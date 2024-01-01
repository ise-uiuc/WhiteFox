
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 32, 2, stride=1, padding=0)
        self.conv_transpose_next = torch.nn.ConvTranspose2d(32, 64, 2, stride=1, padding=0)
        self.conv_transpose_last = torch.nn.ConvTranspose2d(64, 32, 2, stride=1, padding=0)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.adaptive_pool = torch.nn.AdaptiveMaxPool2d(4)
        self.relu = torch.nn.ReLU(inplace=False)
        self.relu6 = torch.nn.ReLU6(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.conv_transpose_next(v5)
        v7 = v6 + 3
        v8 = torch.clamp_min(v7, 0)
        v9 = torch.clamp_max(v8, 6)
        v10 = v9 / 6
        v11 = self.conv_transpose_last(v10)
        v12 = torch.mean(v1, -1, True)
        v13 = torch.mean(v1, -2, True)
        v14 = torch.mean(v1, -1, True)
        v15 = torch.mean(v1, -2, True)
        v16 = torch.std(v1, -1, True)
        v17 = torch.std(v1, -2, True)
        v18 = torch.std(v1, -1, True)
        v19 = torch.std(v1, -2, True)
        v20 = torch.var(v1, -1, True)
        v21 = torch.var(v1, -2, True)
        v22 = torch.var(v1, -1, True)
        v23 = torch.var(v1, -2, True)
        v24 = torch.mean(v1, -1, True) * 1e-05
        v25 = torch.mean(v1, -2, True) * 1e-05
        v26 = torch.mean(v1, -1, True) * 1e-05
        v27 = torch.mean(v1, -2, True) * 1e-05
        v28 = torch.std(v1, -1, True) * 1e-05
        v29 = torch.std(v1, -2, True) * 1e-05
        v30 = torch.std(v1, -1, True) * 1e-05
        v31 = torch.std(v1, -2, True) * 1e-05
        v32 = torch.var(v1, -1, True) * 1e-05
        v33 = torch.var(v1, -2, True) * 1e-05
        v34 = torch.var(v1, -1, True) * 1e-05
        v35 = torch.var(v1, -2, True) * 1e-05
        v36 = self.max_pool(v13) * 1e-05
        v37 = self.adaptive_pool(v1) * 1e-05
        v38 = self.relu(v36)
        v39 = self.relu6(v37) * 1e-05
        v40 = self.sigmoid(v37)
        return v40
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
