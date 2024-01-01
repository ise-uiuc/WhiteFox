
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.5, num_heads=4, d_model=64):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.d_kv = d_model // num_heads
        self.inner_dim = self.d_kv * self.num_heads
        # Matmuls
        self.query_key_matmul = torch.nn.Linear(d_model, self.d_kv, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.output_matmul = torch.nn.Linear(self.inner_dim, d_model, bias=False)
        # Positional bias
        self.positional_bias = torch.nn.Embedding(self.inner_dim, self.inner_dim)

    def forward(self, x1, x2, x3):
        assert x1.shape == x2.shape == x3.shape
        # Attention mask
        n = x1.shape[1]
        r = n % self.d_kv
        mask = torch.eye(n, device=x1.device).bool()
        if r!= 0:
            mask_r = torch.zeros(n, n - r, device=x1.device).bool()
            mask = torch.cat([mask[:, :-r], mask_r], dim=1)
        mask = mask.view(1, 1, n, n)
        # Queries, keys, and values
        q = self.query_key_matmul(x1)
        k = self.query_key_matmul(x2)
        v = self.query_key_matmul(x3)
        # Split into heads
        q = q.contiguous().view(-1, n, self.num_heads, self.d_kv).transpose(1, 2)
        k = k.contiguous().view(-1, n, self.num_heads, self.d_kv).transpose(1, 2)
        v = v.contiguous().view(-1, n, self.num_heads, self.d_kv).transpose(1, 2)
        # Scaled dot product computation
        qk = torch.matmul(q, k.transpose(-2, -1))
        v = v.unsqueeze(1)
        qk = (qk / np.sqrt(self.d_kv)).softmax(dim=-1)
        dropout_qk = self.dropout(qk)
        y = torch.matmul(dropout_qk, v).transpose(1, 2).contiguous()
        y = y.view(-1, self.inner_dim)
        y = self.output_matmul(y)
        return y

# Initializing model
m = Model()

# Inputs to the model (batch, seq_length, d_model)
x1 = torch.randn(2, 9, 3)
x2 = torch.randn(2, 8, 3)
x3 = torch.randn(2, 8, 3)
