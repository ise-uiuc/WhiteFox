
class Model1(torch.nn.Module):
    def __init__(self, num_heads, hidden_size, dropout_p):
        super().__init__()
        self.qkv = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, attn_mask):
        qkv = self.qkv(query)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(q.size(0), q.size(1), num_heads, -1)
        k = k.view(k.size(0), k.size(1), num_heads, -1)
        v = v.view(v.size(0), v.size(1), num_heads, -1)
        qk = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = self.dropout(attn_weight)
        o = attn_weight @ v
        o = o.view(o.size(0), o.size(1), -1)
        return o

# Initializing the model
m = Model1(num_heads, hidden_size, dropout_p)

# Inputs to the model
query = torch.randn(1, src_seq_len, hidden_size)
key = torch.randn(1, tgt_seq_len, hidden_size)
value = torch.randn(1, tgt_seq_len, hidden_size)
attn_mask = torch.randn(1, num_heads, 1, src_seq_len)
