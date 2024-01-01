
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(x3)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, x5)
        v5 = torch.matmul(v4, x4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 3072)
x3 = torch.randn(1, 16, 1, 1).log().softmax(dim=1)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1)
