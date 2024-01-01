
class Model(torch.nn.Module):
    def __init__(self, m,n,o,p):
        super().__init__()
        self.q_ = torch.nn.Linear(m, n)
        self.k_ = torch.nn.Linear(p, o)
    
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        q = self.q_(query).unsqueeze(1)  # [seq_len*batch, 1, dim]
        k = self.k_(key).transpose(-2, -1)  # [seq_len*batch, dim, num_heads]
        qk = torch.matmul(q, k)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(m=4, n=3, o=2, p=4)

# Inputs to the model
query = torch.randn(4, 3)
key = torch.randn(5, 4)
value = torch.randn(5, 2)
inv_scale_factor = torch.rand(1)
dropout_p = torch.rand(1)
