
class Model(torch.nn.Module):
    def forward(self, x):
        v0 = torch.ones(x.shape[0], 2, 2).expand_as(x)
        v1 = torch.cat([x, v0], 1)
        v2 = x.shape[0]
        v3 = torch.zeros([v2, 2, 1, 5], dtype=x.dtype, layout=x.layout, device=x.device)
        v4 = v3
        v5 = x.shape[0]
        v6 = torch.zeros([v5, 2, 1, 14], dtype=x.dtype, layout=x.layout, device=x.device)
        v7 = v6
        v8 = x.shape[0]
        v9 = x.shape[2]
        v10 = x.shape[3]
        v11 = x.shape[0]
        v12 = x.shape[2]
        v13 = x.shape[3]
        v14 = v0.shape[0]
        v15 = v0.shape[1]
        v16 = v0.shape[2]
        v17 = v0.shape[3]
        v18 = torch.rand(v17, dtype=x.dtype, layout=v0.layout, device=v0.device)
        v19 = v18
        v20 = torch.rand(v16, dtype=x.dtype, layout=v0.layout, device=v0.device)
        v21 = v20
        v22 = torch.ones(v13, 1, dtype=v3.dtype, layout=v3.layout, device=v3.device)
        v23 = v22
        v24 = torch.zeros(v10, 1, dtype=v4.dtype, layout=v4.layout, device=v4.device)
        v25 = v24
        v26 = torch.cat([v3, v23.unsqueeze(2).unsqueeze(3)], dim=2)
        v27 = v26
        v28 = torch.cat([v27, v4, v25], dim=3)
        v29 = v28
        v30 = torch.cat([v0, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17], 0)
        v31 = v30
        v32 = torch.cat([v21, v29, v19, v25], dim=3)
        v33 = v32
        v34 = torch.cat([v1, v7, v31, v33], dim=0)
        return v34

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 3, 64, 64)
