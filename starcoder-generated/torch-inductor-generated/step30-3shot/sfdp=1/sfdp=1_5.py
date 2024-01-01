
class Model(torch.nn.Module):
    def __init__(self, dim, n_head, dropout_p):
        super().__init__()
        self.dim_head = dim_head
        self.n_head = n_head
        self.scale_factor = self.dim_head ** -0.5
        self.emb_proj_k = torch.nn.Linear(in_dim, self.dim_head * self.n_head)
        self.emb_proj_q = torch.nn.Linear(in_dim, self.dim_head * self.n_head)
        self.emb_proj_v = torch.nn.Linear(in_dim, self.dim_head * self.n_head)
        self.attention = torch.nn.MultiheadAttention(self.dim_head * self.n_head, self.n_head, dropout_p)
 
    def forward(self, x_emb_k, x_emb_q, x_emb_v):
        q = self.emb_proj_q(x_emb_q)
        k = self.emb_proj_k(x_emb_k)
        v = self.emb_proj_v(x_emb_v)
        q, k, v = q.reshape(B, self.n_head, s_len, self.dim_head), k.reshape(B, self.n_head, s_len, self.dim_head), v.reshape(B, self.n_head, s_len, self.dim_head)
        q, k, v = q.transpose(2,1), k.transpose(2,1), v.transpose(2,1)
        q, k, v = q * self.scale_factor, k, v
        attn, output = self.attention(q, k, v)
        attn = attn.transpose(2, 1)
        output = output.reshape((B, nhead * self.dim_head, s_len))
        output = self.dropout(output)
        return output

# Initializing the model
d_model = 128
n_head = 4
dropout_p = 0.2
m = Model(d_model, n_head, dropout_p)

# Inputs to the model
x_emb_k = torch.randn(4, N, d_model)
x_emb_q = torch.randn(4, N, d_model)
x_emb_v = torch.randn(4, N, d_model)
