
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(x3)
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, x4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(6, 4, 10)
x2 = torch.randn(6, 10, 4)
x3 = torch.randint(1, 10, [6, 4])
x4 = torch.randn(6, 4, 5)
