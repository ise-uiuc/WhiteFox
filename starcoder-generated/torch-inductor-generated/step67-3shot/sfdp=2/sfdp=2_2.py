
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(100)
 
    def forward(self, x1, x2, x3):
        v1 = self.norm(x1)
        v2 = torch.matmul(x2, x3.transpose(-2, -1))
        v3 = v2.div(2.6)
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.1)
        v6 = v5.matmul(x1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 100)
x2 = torch.randn(4, 30, 100)
x3 = torch.randn(4, 100, 1)
