
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, x1, x2):
        v = torch.matmul(x1, x2.transpose(-2, -1))
        v1 = v / 0.1
        v2 = v1.softmax(dim=-1)
        v3 = self.dropout(v2)
        o = v3.matmul(x2)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 2, 64)
x2 = torch.randn(1, 3, 3, 64)
