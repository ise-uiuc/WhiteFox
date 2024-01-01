
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.flatten(start_dim=1)
        v2 = torch.zeros([1, 1, 64*64])
        for i in range(64*64):
            v3 = v2[:, :, i:i+512]
            a, b, c, d, e = v3.shape
            v3 = v1[:, i].unsqueeze(-1).repeat([a, b, c, d, e])
            v3 = v3 * (i + 1)
            v2[:, :, i:i+512] = v2[:, :, i:i+512] + v3
        v4 = v2.transpose(1, 2)
        v5 = v2 * 0.9544624663838703
        v6 = v4 * 0.8617381228098389
        v7 = v6 * 0.5
        v8 = torch.tanh(v7)
        v9 = v8 * v8
        v10 = v9 * 0.35935407278928806
        v11 = v2 * 0.7828651177241366
        v12 = v11 * 0.20489670199080818
        v13 = v2 * 0.686066355019853
        v14 = v13 * v13
        v15 = v14 * 0.015417369689287388
        v16 = torch.tanh(v12)
        v17 = v13 * v16
        v18 = v17 * 0.9647933044650485
        v19 = v12 * 0.9771225998543117
        v20 = v12 * v15
        v21 = v20 / 4096
        v22 = v21 * 310.79436151793565
        v23 = v21 * v19
        v24 = v23 + v22
        v25 = v14 + v18
        v26 = v23 * 0.5904509821971973
        v27 = torch.tanh(v25)
        v28 = torch.tanh(v24)
        v29 = v27 + v28
        v30 = v27 * 0.8669650154418945
        v31 = v28 + v30
        v32 = v27 + v31
        v33 = v27 * 0.13012893318236604
        v34 = v28 + v33
        v35 = v28 + v34
        v36 = v27 * 0.3967544633565598
        v37 = torch.tanh(v31)
        v38 = torch.tanh(v32)
        v39 = v37 + v38
        v40 = v37 * 0.979462285583588
        v41 = torch.tanh(v35)
        v42 = torch.tanh(v36)
        v43 = v41 * 0.6759555794346873
        v44 = v42 + v43
        v45 = v37 + v44
        v46 = v41 + v44
        v47 = v42 * 0.5175347369939032
        v48 = v45 + v47
        v49 = v45 * 0.7472524297372166
        v50 = v46 + v49
        v51 = v46 + v50
        v52 = torch.tanh(v51)
        v53 = v52 + v52
        v54 = v50 + v52
        return v54
# Inputs to the model
x1 = torch.randn(1, 64, 64)
