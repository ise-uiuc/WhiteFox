
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, d_embed, dropout_p=0.0):
        super().__init__()
        self.n_head = n_head
        self.d_embed = d_embed
        self.dropout_p = dropout_p
        self.q_proj = torch.nn.Linear(d_embed, d_embed)
        self.k_proj = torch.nn.Linear(d_embed, d_embed)
        self.v_proj = torch.nn.Linear(d_embed, d_embed)
        self.n_proj = torch.nn.Linear(d_embed, d_embed)
 
    def forward(self, query, key, value, mask=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        n = self.n_proj((q + k + v) / self.n_head)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = self.n_head ** -0.5
        scaled_qk = qk * scale_factor
 
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
 
        output = dropout_qk.matmul(v)
        return output
 
class Model(torch.nn.Module):
    def __init__(self, n_head, d_embed, dropout_p):
        super().__init__()
        self.mha = MultiHeadAttention(n_head, d_embed, dropout_p)
 
    def forward(self, query, key, value, mask=None):
        output = self.mha(query, key, value, mask)
        return output

# Initializing the model
n_head = 2
d_embed = 16
dropout_p = 0.0
m = Model(n_head, d_embed, dropout_p)

# Inputs to the model
query = torch.randn(1, 16, d_embed)
key = torch.randn(1, 16, d_embed)
value = torch.randn(1, 16, d_embed)
