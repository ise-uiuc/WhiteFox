
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        t = torch.matmul(q, k.transpose(-2, -1))
        s = t * scale_factor
        a = s.softmax(dim=-1)
        d = torch.nn.functional.dropout(a, p=dropout_p)
        x = torch.matmul(d, v)
        return x

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 512, 128)
k = torch.randn(1, 512, 64)
v = torch.randn(1, 512, 64)
scale_factor = 0.12
dropout_p = 0.2
