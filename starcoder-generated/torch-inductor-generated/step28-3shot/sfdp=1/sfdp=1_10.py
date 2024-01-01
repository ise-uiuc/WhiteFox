
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        mm = torch.matmul(x1, x2.transpose(-2, -1))
        v1 = mm.div(0.1)
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=0.95)
        v4 = v3.matmul(x1)
        return v3, v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 32)
x2 = torch.randn(1, 64, 64)
