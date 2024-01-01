
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, d_p):
        v1 = torch.matmul(x1, x2.transpose(-2, -1), dim=-1)
        v2 = v1.div(0.5)
        v3 = torch.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=d_p, training=False)
        v5 = v4.matmul(x2)
        return v5
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 64)
x2 = torch.randn(2, 64, 100)
d_p = 0.75
