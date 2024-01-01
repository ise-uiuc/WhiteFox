
def make_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
            self.linear1 = torch.nn.Linear(1000, 1000)
            self.linear2 = torch.nn.Linear(1000, 10)

        def forward(self, x1):
            v1 = self.conv(x1)
            v2 = self.linear1(v1)
            v3 = v2 * 0.5
            v4 = v2 * 0.7071067811865476
            v5 = self.linear2(v4)
            v6 = v3 * v5
            v7 = v5 * 0.5
            v8 = v5 * 0.8
            v9 = v5 * 0.3
            v10 = v5 * 0.6
            v11 = v5 * 0.2738612787525831
            v12 = torch.erf(v10)
            v13 = torch.erf(v4) + 1
            v14 = torch.erf(v11)
            v15 = torch.erf(v9)
            v16 = x1.size(3)
            v17 = x1.size(3)
            v18 = x1.size(3)
            v19 = x1.size(3)
            v20 = x1.size(2)
            v21 = x1.view(x1.size(0), x1.size(1), 8, 8, -1, 1, 1)
            v22 = torch.sum(v21, 4)
            v23 = torch.sum(v22, 1)
            v24 = torch.sum(v23, 2)
            v25 = v24 + v18
            v26 = torch.view(v1, (1, 10000000,))
            v27 = torch.nn.functional.softmax(v26)
            v28 = torch.view(v27, (-1, 10, 1000,))
            v29 = torch.mean(v28, 1)
            v30 = torch.sum(v29, 1)
            v31 = x1.size(3)
            v32 = x1.size(2)
            v33 = x1.view(x1.size(0), -1, 1, 1)
            v34 = torch.cat((v6, v13,), 1)
            v35 = torch.unsqueeze(v34, 2)
            v36 = torch.cat((v35, v35,), 2)
            v37 = torch.unsqueeze(v36, 3)
            v38 = torch.permute(v37, (0, 2, 3, 1, 4, 5,))
            v39 = v38 + v38
            v40 = torch.permute(v39, (0, 2, 1, 3, 4, 5, 6,))
            a1 = 36
            v41 = 17 * a1 + 78 * v19
            v42 = torch.view(v12, (v12.size(0), v41,))
            a2 = 45
            v44 = 37 * a2 + 89 * v18
            v45 = torch.view(v30, (v30.size(0), v44,))
            a3 = 49
            v46 = 39 * a3 + 10 * v18
            v47 = torch.view(v25, (v25.size(0), v46,))
            a4 = 55
            v48 = 49 * a4 + 1 * v18
            v49 = torch.view(v31, (v31.size(0), v48,))
            a5 = 60
            v50 = 26 * a5 + 47 * v32
            v51 = 0.0
            v52 = torch.zeros([x1.size(0), v50, v32, v17,], dtype=v1.dtype,)
            v53 = torch.empty([x1.size(0), v50, v33, v17,], dtype=v1.dtype,)
            v54 = torch.rand([x1.size(0), v50, v33, v17,], dtype=v1.dtype,)
            v55 = v52 + v55
            a6 = 644
            a7 = 645
            v57 = (torch.tensor(1, dtype=v1.dtype, device=torch.device('cuda'),) < torch.tensor(0.5, dtype=v1.dtype, device=torch.device('cuda'),)).int()
            v58 = torch.view(v57, (x1.size(0), 1, v16, v16,))
            v59 = v1 + v59
            v60 = v58 * v59
            v61 = v1 * v60
            v62 = v61 / v61
            v63 = v61 * v61
            v64 = v63 * 2
            v65 = v63 * 3
            v66 = v63 * v63
            v67 = v66 * 5
            v68 = v4+a6;v69 = v4+a7;torch.sum(v68, 3);
            return v63

# Initializing the model
