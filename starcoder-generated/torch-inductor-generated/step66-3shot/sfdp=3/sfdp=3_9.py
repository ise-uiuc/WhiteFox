
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k1, v1, scale_factor=8, dropout_p=0.9):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        v1 = qk * (1 / scale_factor)
        v3 = v1.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        return torch.matmul(v4, v1)

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(3, 12, 10)
k1 = torch.randn(6, 14, 10)
v1 = torch.randn(5, 14, 11)
