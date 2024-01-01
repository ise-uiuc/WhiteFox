
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self. ScaledDotProductAttention()
 
    def ScaledDotProductAttention(self, query, key, value, scale_factor=1, dropout_p=0.0):
        q_s = query.shape[:-1]
        k_s = key.shape[:-1]
        v_s = value.shape[:-1]
        assert (q_s[:-1] == k_s[:-1])
        assert (v_s[:-1] == k_s[:-1])
        head_dim = q_s[-1]
        head = torch.matmul(query.reshape(-1, head_dim), key.transpose(-2, -1).reshape(-1, head_dim))
        scaled_head = head.mul(scale_factor)
        softmax_head = scaled_head.softmax(dim=-1).reshape(*q_s, *v_s, head_dim)
        doutput = torch.nn.functional.dropout(softmax_head, p=dropout_p)
        output = doutput.matmul(value)
        return output
 
    def forward(self, x):
        nbatch, seq_max, embed_dim = x.shape
        qkv = [l(x).view(nbatch, seq_max, self.h, self.d_k).transpose(1, 2) for l in self.linears[:-1]]
        q, k, v = qkv
        scale_factor = self.d_k ** -0.5
        o = self.ScaledDotProductAttention(q, k, v, scale_factor)
        o = o.transpose(1, 2).contiguous().view(nbatch, seq_max, self.h * self.d_k)
        o = self.linears[-1](o)
        return o

# Initializing the model
m = MultiHeadAttention(2, 2)

# Inputs to the model
x1 = torch.randn(1, 4, 2)
