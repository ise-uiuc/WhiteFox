
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2)
        v2 = v1 / math.sqrt(v1.size(-1))
        v3 = v2 + x3
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.matmul(v4, x2)
        return v5
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 100)
x2 = torch.randn(1, 32, 100)
x3 = torch.rand(1, 100, 100)
