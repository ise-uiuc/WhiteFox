
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.25)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(1, 2))
        v2 = v1
        v3 = v1.div(dim=1)
        v4 = v2.softmax(dim=0)
        v5 = v4
        v6 = v3.expand(v3.shape)
        v7 = torch.matmul(v6, v5)
        t1 = v7
        v8 = t1
        t2 = v8
        v9 = t2
        v10 = self.dropout(v9)
        t3 = v10
        return t3


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 5, 16)
x2 = torch.randn(32, 16, 12)
