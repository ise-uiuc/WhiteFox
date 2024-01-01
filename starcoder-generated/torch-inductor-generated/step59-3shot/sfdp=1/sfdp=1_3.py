
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        t1 = torch.matmul(x1, x2.transpose(-2, -1))
        v1 = t1.div(0.004569929859719439)
        v2 = torch.nn.functional.softmax(v1, dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=0.1767165779685974)
        v4 = v3.matmul(x3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 3072, 4)
x2 = torch.randn(128, 3072, 4)
x3 = torch.randn(128, 4, 64)
x4 = torch.randn(128, 4, 64)
