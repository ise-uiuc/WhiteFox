
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.gt(0)
        v3 = v2.int()
        v4 = v3.add(2)
        v5 = v4.long()
        v6 = v5.sub(1)
        v7 = v6!= 2
        v8 = v7 | v2
        v9 = v8.float()
        v10 = v8.lt(0)
        v11 = v10 - 0.5
        v12 = v9 & v11
        v13 = v12.sum(0)
        v14 = v13.exp()
        output = v14.log()
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
