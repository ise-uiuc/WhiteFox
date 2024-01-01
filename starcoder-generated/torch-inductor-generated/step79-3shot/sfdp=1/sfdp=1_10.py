
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(0.0625)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.8)
        v3.matmul(x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 64, 1)
x2 = torch.randn(32, 1, 49)
x3 = torch.randn(32, 49, 64)
