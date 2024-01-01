
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(1.0)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, 0.25)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
x1[1, 1, 1, 2] = -1.0
x2 = torch.randn(1, 3, 128, 128)
