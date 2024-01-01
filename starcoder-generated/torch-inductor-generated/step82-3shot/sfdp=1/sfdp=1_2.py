
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(x3)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=x4)
        v5 = torch.matmul(v4, x5)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
x2 = torch.randn(1, 4, 4)
x3 = torch.randn(1)
x4 = torch.randn(1)
x5 = torch.randn(1, 4, 4)
