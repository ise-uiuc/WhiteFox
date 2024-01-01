
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 5.0
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.25)
        v5 = v4.matmul(x3)
        v6 = v5.matmul(x4.transpose(-2, -1))
        return v6

# Initializing the model
x = []
for _ in range(4):
    x.append(torch.randn(20, 64, 64))
m = Model()

# Inputs to the model
