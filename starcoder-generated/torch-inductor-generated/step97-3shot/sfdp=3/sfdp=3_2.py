
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.matmul(x1.transpose(-2, -1))
        v2 = v1.mul(10000)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.0)
        return v4.matmul(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
