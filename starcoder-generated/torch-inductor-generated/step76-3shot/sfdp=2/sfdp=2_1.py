
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        t1 = torch.matmul(x1, x2.T)
        v1 = t1 / 32.123456789
        v2 = F.softmax(v1, dim=-1)
        v3 = F.dropout(v2, p=0.2)
        v4 = torch.matmul(v3, x2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 11)
x2 = torch.randn(1, 11, 8)
