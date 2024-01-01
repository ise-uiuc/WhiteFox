
class Model(torch.nn.Module):
    def __init__(self, q_dim, k_dim):
        super().__init__()
        self.q = torch.nn.Linear(q_dim, k_dim)
        self.k = torch.nn.Linear(k_dim, k_dim)
 
    def forward(self, query, value, key, dropout_p):
        inv_scale_factor = torch.rsqrt(torch.tensor(query.size(-1)).float())
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(q_dim=3, k_dim=3)

# Inputs to the model
query = torch.randn(1, 2, 3)
value = torch.randn(1, 2, 3)
key = torch.randn(1, 2, 3)
dropout_p = torch.tensor(0.5)
