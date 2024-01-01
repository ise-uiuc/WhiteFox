
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x2, x3):
        v2 = torch.matmul(x2, x3.transpose(-2, -1))
        v3 = v2.mul(2.0)
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.1, training=True)
        v6 = v5.matmul(x3)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(3, 4, 5)
x3 = torch.randn(3, 5, 6)
