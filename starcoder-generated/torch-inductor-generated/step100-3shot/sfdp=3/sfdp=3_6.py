
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.matmul(x1, x1.transpose(-2, -1))
        v2 = v1 * 50
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = torch.matmul(v4, x1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 256)
x2 = torch.randn(1, 16, 1, 1)
