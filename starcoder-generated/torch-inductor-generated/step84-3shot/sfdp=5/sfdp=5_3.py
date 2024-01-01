
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 128
        self.seq_len = 256
        self.dim = 4096 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
batch_size = 1
seq_len = 20
dim_m = 2048
dim_v = 4096
heads = 48
query = torch.randn(batch_size, heads, seq_len, dim_m)
key = torch.randn(batch_size, heads, seq_len, dim_m)
value = torch.randn(batch_size, heads, seq_len, dim_v)
attn_mask = torch.eye(seq_len).to(dtype=query.dtype)
attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
attn_mask = attn_mask.repeat(batch_size, 1, 1, 1)
