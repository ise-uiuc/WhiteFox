
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = x1.shape
        v2 = torch.Tensor.size(x1, 0)
        v3 = torch.Tensor.size(x1, 1)
        v4 = torch.Tensor.size(x1, 2)
        v5 = torch.Tensor.size(x1, 3)
        v6 = x1.permute(0, 1, 3, 2)
        v7 = x1.permute(0, 3, 2, 1)
        v8 = torch.Tensor.matmul(v6, v7)
        v9 = torch.Tensor.resize_([v8], [v2, v2 * v3, v5, v5])
        v10 = torch.Tensor.permute(v9, 0, 2, 1, 3)
        v11 = v10.reshape((v2, v4 * v3, v4 * v3))
        v12 = torch.Tensor.reshape(v11, (v4, v3, v4, v3))
        v13 = v12.permute(0, 2, 1, 3)
        v14 = v13.reshape((v2 * v3, v4 * v3))
        v15 = torch.Tensor.matmul(v14, x1)
        v16 = torch.Tensor.sign(v15)
        v17 = 1 / v16
        v18 = self.negative_slope * v17
        v19 = self.negative_slope
        v20 = v15.shape
        v21 = torch.Tensor.size(v15, 0)
        v22 = torch.Tensor.size(v15, 1)
        v23 = v20[0]
        v24 = v22[0]
        v25 = v21 * v23
        v26 = v25.double()
        v27 = v15.int()
        v28 = v27 > 0
        v29 = v28.float()
        v30 = v18 * v29
        v31 = v29 <= 0
        v32 = torch.Tensor.where(v31, v19, v29)
        v33 = v32[0]
        v34 = v32[1]
        v35 = v30 <= v33
        v36 = 0
        v37 = v30 >= v34
        v38 = 0
        v39 = v30!= v34
        v40 = v31.float()
        v41 = v40 * v37
        v42 = (0.0 - v34) * v41
        v43 = v39 * v36
        v44 = v30 - v33
        v45 = v30 - v34
        v46 = v33 - v34
        v47 = v35 * v36
        v48 = v35 * v42
        v49 = v47 + v48
        out = v46 * v44 * v45 + v43 * v49
        return out

# Initializing the model
negative_slope = 0.01
m = Model(negative_slope)

# Inputs to the model
x1 = torch.randn(2, 2, 4, 4)
