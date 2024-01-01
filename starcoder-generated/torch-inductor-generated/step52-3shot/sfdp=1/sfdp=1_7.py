
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(0.030620265117591095)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.2)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 256, 3)
x2 = torch.randn(12, 256, 256)
x3 = torch.randn(12, 256, 100)
