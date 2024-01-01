
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.query_scaling = torch.nn.Parameter(torch.ones(1, 1, query_dim))
 
    def forward(self, q, k, v, dropout_p):
        q = q * self.query_scaling
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.tensor(query_dim)).float()
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(query_dim, key_dim, value_dim)

# Inputs to the model
q = torch.randn(1, 1, query_dim)
k = torch.randn(1, key_dim, query_dim)
v = torch.randn(1, value_dim, query_dim)
