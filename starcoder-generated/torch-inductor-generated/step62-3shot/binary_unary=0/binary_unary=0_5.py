
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.matmul(x1, x1)
        v2 = torch.matmul(x1, x1)
        v3 = self.conv1(x1)
        v4 = v3 + x1
        v5 = torch.argmin(v4)
        v6 = self.conv1(v4)
        v7 = v7 + v6
        v8 = v7 - v1
        v9 = torch.relu(v8)
        v10 = torch.sigmoid(v9[..., v5])
        v11 = v10 * v5
        v12 = v11[0,...]
        v13 = v7.sum(-4)
        v14 = v12 - v13
        v15 = self.conv1(x1)
        v16 = v14[..., v5].sum()
        v17 = v16 * v15
        v18 = torch.tanh(v17)
        v19 = v18 + v17
        v20 = v19.matmul(v19)
        v21 = v20.matmul(self.conv1(x1))
        v22 = torch.nn.functional.max_pool2d(v21, 7, stride=2, padding=3)
        v23 = v22.view(-1)
        v24 = self.conv1(x1)
        v25 = torch.mm(v24, v24)
        v26 = torch.mm(v24, v2)
        v27 = torch.mm(v19, v24)
        v28 = v27[0, 1]
        v29 = v28 + v19
        v30 = v25.mean()
        v31 = v23.var()
        v32 = v26[..., 1]
        v33 = torch.max(v32, v30)
        v34 = v33 + v32
        v35 = torch.nn.functional.softmax(v7)
        v36 = v35.argmax()
        v37 = torch.nn.functional.max_pool2d(x1, 3)
        v38 = torch.softmax(v14, dim=0)
        v39 = v38[..., v5]
        v40 = v39 + v14
        v41 = v40.topk(3)
        v42 = torch.nn.functional.conv2d(x1, weight=torch.empty((40, 40, 1, 1), device='cuda:0', dtype=torch.float, requires_grad=True), bias=None, stride=1, padding=0, dilation=1, groups=1)
        v43 = v42[0, 0, 0]
        v44 = v41[0][1].matmul(v7)
        v45 = v43 / v37[0, 0, 0]
        v46 = v17.permute(0, 1).unsqueeze(0)
        v47 = v46.unsqueeze(-1).permute(1, 2, 0)
        v48 = v47 - torch.empty((2, 1), device='cuda:0', dtype=torch.float)
        v49 = v48 > 0
        v50 = v49.all(dim=0)[..., 0]
        v51 = v50[0]
        v52 = v19[0, 1, v36]
        v53 = v25[..., 0]
        v54 = torch.sqrt(v40).sum(dim=0)
        v55 = v19 - v27
        v56 = v55 / v24
        v57 = v45.abs()
        v58 = torch.abs(v28)
        v59 = v14.mul_(v27, inplace=False)
        v60 = v59[..., v5]
        v61 = v60 * v18
        v62 = v61.mean()
        v63 = v52 + v54
        v64 = v63 + v44
        v65 = v19.permute(0, -1) / v31
        v66 = v65.permute(1, 0)
        v67 = torch.norm(v25)
        return v57
# Inputs to the model      
x1 = torch.randn(1, 40, 40, 40)
