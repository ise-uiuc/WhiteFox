
class Model(torch.nn.Module):
    def __init__(self, inv_scale_dim, dropout_p):
        super().__init__()
        self.i_s = inv_scale_dim
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_s = torch.rsqrt(torch.tensor([qk.size()[-1]]))
        s_qk = qk.div(inv_s.to(qk))
        do_qk = torch.nn.functional.dropout(s_qk, dropout_p)
        out = do_qk.matmul(value)
        return out

# Initializing the model
model = Model(3, 0.0)

# Inputs to the model
batch, seq, length, size = 2, 3, 4, 5
query = torch.randn(batch, seq, length, size)
key = torch.randn(batch, seq, length, size)
value = torch.randn(batch, seq, length, size)
dropout_p = 0.8
