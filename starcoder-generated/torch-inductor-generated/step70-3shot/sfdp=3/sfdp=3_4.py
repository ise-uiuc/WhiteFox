
class Model(torch.nn.Module):
    def __init__(self, dim_k, dim_v, dim_q, dropout_p):
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        self.dropout_p = dropout_p
        self.dropout2d = torch.nn.Dropout2d(dropout_p)
 
    def forward(self, q, k, v):
        s = 1 / math.sqrt(self.dim_k)
        t1 = torch.matmul(q, k.transpose(-2, -1))
        t2 = t1 * s
        t3 = t2.softmax(dim=-1)
        t4 = self.dropout2d(t3)
        output = t4.matmul(v)
        return output

# Initializing the model
dim_k = 64
dim_v = 64
dim_q = 64
dropout_p = 0.1
m = Model(dim_k, dim_v, dim_q, dropout_p)

# Inputs to the model
q = torch.randn(1, 1, dim_q, dim_k)
k = torch.randn(1, 1, dim_v, dim_k)
v = torch.randn(1, 1, dim_v, dim_v)
