
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v2 = torch.matmul(x1, x2.transpose(-1, -2))
        v4 = v2.div(0.5)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.001)
        v1 = torch.matmul(v6, x3)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 5)
x2 = torch.randn(2, 5, 6)
x3 = torch.randn(2, 4, 6)
x4 = torch.randn(2, 6, 8)
