
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x3, x4):
        v1 = torch.matmul(x3, x4.transpose(-2, -1))
        v2 = v1.div(1e-12)
        v3 = torch.nn.functional.dropout(v2,.65, False)
        v4 = v3.matmul(x4)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 4, 8)
x4 = torch.randn(1, 5, 8)
