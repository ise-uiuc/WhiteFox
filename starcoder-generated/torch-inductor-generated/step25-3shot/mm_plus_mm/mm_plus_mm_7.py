
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.randn(3, 3)
        self.w2 = torch.randn(3, 3)
    def forward(self, x1, x10, x2, x3, x5, x6, x8, x9):
        v1 = torch.mm(x10, self.w1)
        v2 = torch.mm(x10, self.w2)
        v3 = torch.mm(x2, self.w1)
        v4 = torch.mm(x2, self.w2)
        v5 = torch.mm(x5, self.w1)
        v6 = torch.mm(x5, self.w2)
        v7 = torch.mm(x6, self.w1)
        v8 = torch.mm(x6, self.w2)
        v9 = torch.mm(x9, self.w1)
        v10 = torch.mm(x9, self.w2)
        v11 = torch.mm(v1, v3)
        v12 = torch.mm(v1, v4)
        v13 = torch.mm(v1, v13) + v11
        v14 = torch.mm(v10, v3)
        v15 = torch.mm(v10, v4)
        v16 = torch.mm(v10, torch.mm(input1, torch.mm(torch.mm(self.w1, torch.mm(input1, v3)), v16))) + v14 + v14
        v17 = torch.mm(v6, self.w1)
        v18 = torch.mm(v6, self.w2)
        v19 = torch.mm(v8, self.w1)
        v20 = torch.mm(v8, self.w2)
        v21 = torch.mm(v19, v3)
        v22 = torch.mm(v19, v4)
        v23 = torch.mm(v22, v3)
        return v5 + v23 + v13 + v15 + v21 + v22
# Inputs to the model
x1 = torch.randn(4, 2)
x10 = torch.randn(4, 2)
x2 = torch.randn(2, 4)
x3 = torch.randn(2, 3)
x5 = torch.randn(4, 5)
x6 = torch.randn(4, 2)
x8 = torch.randn(2, 4)
x9 = torch.randn(2, 5)
