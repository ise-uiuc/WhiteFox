
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k2, v3):
        v1 = torch.matmul(q1, k2.transpose(-2, -1))
        v2 = v1 * 1./((v1.shape[-1])**0.5)
        v4 = torch.nn.functional.softmax(v2, dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.9)
        v6 = torch.matmul(v5, v3)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 3, 64, 64)
k2 = torch.randn(1, 4, 64, 64)
v3 = torch.randn(1, 4, 64, 64)
