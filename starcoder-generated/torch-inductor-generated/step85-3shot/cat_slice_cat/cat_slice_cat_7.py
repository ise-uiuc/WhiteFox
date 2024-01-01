
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1):
        v0 = x1.size(1)
        l1 = list(x1.size())
        v2 = l1
        v2[1] = 9223372036854775807
        t0 = torch.slice(None, None, None, None, None, size, None, (v0,), False, (v2,), False)
        l3 = list(x1.size())
        v4 = l3[1]
        v5 = self.size
        l4 = list(x1.size())
        v6 = l4[1]
        v7 = v6
        v7 += v5
        l5 = list(x1.size())
        v8 = l5[1]
        v9 = v8
        v9 -= v5
        v14 = v9
        l6 = list(x1.size())
        v10 = l6[1]
        v11 = self.size
        v12 = v10
        v12 -= v11
        v13 = v11
        v13 *= v12
        l7 = list(x1.size())
        v15 = l7
        v15[1] = size
        t1 = torch.slice(None, None, None, None, None, (9223372036854775807), None, (v14,), False, (v15,), False)
        t2 = torch.concat([x1, t1], 1)
        t3 = torch.slice(None, None, None, None, None, size, None, (v4,), False, (v6,), False)
        l9 = list(v2)
        l9[1] = v7
        t4 = torch.slice(None, None, None, None, None, (9223372036854775807), None, (v13,), False, (l9,), False)
        t5 = torch.concat([t2, t3, t4], 1)
        return t5

# Initializing the model
m = Model(5000)

# Inputs to the model
x1 = torch.randn(12, 30000, 4)
