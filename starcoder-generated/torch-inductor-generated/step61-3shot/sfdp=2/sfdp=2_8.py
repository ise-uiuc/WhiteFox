
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, p):
        i_ = q.matmul(k.transpose(1, 2)) / p
        o = torch.nn.functional.dropout(torch.nn.functional.softmax(i_, dim=-1), p=p).matmul(v)
        return o


# Initializing the model
m = Model()

# Inputs to the model
n = 3
p = 2.4
q = torch.randn(1, 8, 128, 64)
k = torch.randn(1, 8, 128, 64 * 2)
v = torch.randn(1, 8, 128, 64 * 2)
