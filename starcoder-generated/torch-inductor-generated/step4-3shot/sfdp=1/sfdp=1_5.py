
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_head, dropout):
        super().__init__()
        self.qkv_linear = torch.nn.Linear(query_dim, query_dim * 3)
        self.linear = torch.nn.Linear(query_dim * num_head, value_dim)

    def forward(self, x1):
        v1 = self.qkv_linear(x1)
        q, k, v = v1[:, :query_dim], v1[:, query_dim:query_dim * 2], v1[:, query_dim * 2:]
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.tensor(query_dim).float())
        softmax_qk = torch.nn.functional.dropout(qk * inv_scale_factor, p=dropout)
        softmax_qk = torch.nn.functional.softmax(softmax_qk, dim=-1)
        output = torch.matmul(softmax_qk, v)
        final_v = self.linear(output.transpose(1, 2))
        return final_v

# Initializing the model
m = Model(query_dim, key_dim, value_dim, num_head, dropout)

# Inputs to the model
x1 = torch.randn(2, query_dim)
output = m(x1)
