
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * x3
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=x4)
        output = v4.matmul(x2)
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
x1   = torch.randn(1, 100, 8)
x2   = torch.randn(1, 100, 8)
