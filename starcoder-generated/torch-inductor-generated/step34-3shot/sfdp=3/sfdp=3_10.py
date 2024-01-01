
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, k, q, v, scale_factor, dropout_p):
        z = torch.matmul(q, k.transpose(-2, -1))
        z = z * scale_factor
        m = nn.Softmax(dim=-1)
        m = m(z)
        m = torch.nn.functional.dropout(m, p=dropout_p)
        z = m.matmul(v)
        return z

# Initializing the model
m = Model()

# Inputs to the model
k = torch.randn(5, 5)
q = torch.randn(5, 5)
v = torch.randn(5, 10)
scale_factor = 1
dropout_p = 0.1
