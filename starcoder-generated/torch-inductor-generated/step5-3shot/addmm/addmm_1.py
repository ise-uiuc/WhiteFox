
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, t1, t2, t3, t4, t5):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        v3 = torch.mm(x2, x1)
        v4 = torch.mm(x1, x2)
        v5 = v1 + inp
        v6 = torch.mm(x2, x1)
        v7 = torch.mm(x1, v1)
        v8 = torch.mm(x2, v1)
        v9 = torch.mm(x2, v1)
        v10 = v1 + inp
        v11 = torch.mm(x2, x1)
        v12 = torch.mm(x1, x2)
        v13 = v1 + inp
        v14 = torch.mm(x2, x1)
        v15 = torch.mm(v1, v2)
        v16 = t1 + inp
        v17 = torch.mm(v1, v2)
        v18 = v1 + inp
        v19 = torch.mm(v1, v2)
        v20 = t1 + inp
        v21 = t1 + inp
        v22 = v1 + inp
        v23 = torch.mm(v1, v2)
        v24 = torch.mm(t1, t2)
        v25 = v1 + inp
        v26 = torch.mm(x1, x2)
        v27 = t3 + inp
        v28 = t4 + inp
        v29 = torch.mm(x1, v1)
        v30 = torch.mm(x2, v1)
        v31 = torch.mm(x2, v1)
        v32 = t1 + inp
        v33 = torch.mm(x2, x1)
        v34 = v1 + inp
        v35 = torch.mm(v1, v2)
        v36 = t5 + inp
        v37 = torch.mm(v1, v2)
        v38 = v1 + inp
        v39 = torch.mm(t1, t2)
        v40 = torch.mm(v1, v2)
        return v36
# Inputs to the model
x1 = torch.randn(666, 666)
x2 = torch.randn(666, 666)
inp = torch.randn(666, 666)
t1 = torch.randn(666, 666)
t2 = torch.randn(666, 666)
t3 = torch.randn(666, 666)
t4 = torch.randn(666, 666)
t5 = torch.randn(666, 666)
