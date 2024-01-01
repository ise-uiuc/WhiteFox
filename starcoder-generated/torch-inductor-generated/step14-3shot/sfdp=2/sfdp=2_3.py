
class Model(torch.nn.Module):
    def __init__(self, d_query, d_key, d_value, d_out, n_heads, scale_factor, drop_p):
        super().__init__()
        self.dropout = torch.nn.Dropout(drop_p)
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.wq = torch.nn.Parameter(torch.zeros(size=(d_query, d_out)))
        self.wk = torch.nn.Parameter(torch.zeros(size=(d_key, d_out)))
        self.wv = torch.nn.Parameter(torch.zeros(size=(d_value, d_out)))
        torch.nn.init.xavier_normal_(self.wq)
        torch.nn.init.xavier_normal_(self.wk)
        torch.nn.init.xavier_normal_(self.wv)
        self.scale_factor = scale_factor
 
    def forward(self, query, key, value):
        q = torch.matmul(query, self.wq)
        k = torch.matmul(key, self.wk)
        v = torch.matmul(value, self.wv)
        q = q.reshape(q.shape[:-1] + (self.n_heads, self.head_dim))
        k = k.reshape(k.shape[:-1] + (self.n_heads, self.head_dim))
        v = v.reshape(v.shape[:-1] + (self.n_heads, self.head_dim))
        q = q.permute((0, 2, 1, 3))
        k = k.permute((0, 2, 1, 3))
        v = v.permute((0, 2, 1, 3))
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.empty_like(qk).fill_(1 / self.scale_factor)
        qk = qk * inv_scale_factor
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        attended = torch.matmul(dropout_qk, v)
        attended = attended.permute((0, 2, 1, 3))
        attended = attended.reshape(attended.shape[:-2] + (self.n_heads * self.head_dim,))
        return attended

# Initializing the model
m = Model(d_query=26, d_key=26, d_value=26, d_out=36, n_heads=6, scale_factor=994.359619140625, drop_p=0.1)

# Inputs to the model
query = torch.randn(1, 10, 26)
key = torch.randn(1, 8, 26)
value = torch.randn(1, 8, 26)
