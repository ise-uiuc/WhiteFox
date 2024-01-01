
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        r1 = torch.matmul(q, k.transpose(-2, -1))
        r2 = r1 * scale_factor
        r3 = torch.nn.functional.softmax(r2, dim=-1)
        r4 = torch.nn.functional.dropout(r3, p=dropout_p)
        r5 = r4.matmul(v)
        return r5
 
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 32, 64)
k = torch.randn(1, 32, 64)
v = torch.randn(1, 32, 64)
scale_factor = torch.randn(1, 1)
dropout_p = 0.5
