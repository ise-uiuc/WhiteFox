
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v4 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = __//__
        v3 = v4.__//__
        v5 = v3.softmax(dim=-1)
        v6 = v5.__//__
        v7 = torch.matmul(v6, x3)
        return v7
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(3, 10)
x2 = torch.randn(6, 10)
x3 = torch.randn(6, 20)
