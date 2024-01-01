
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_token_len = 256
        self.heads = 32
        self.seq_len = 1024
        self.emb_dim = 301
        self.dim = 800 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.5, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 32, 1024, 800)
key = torch.randn(1, 32, 1024, 800)
value = torch.randn(1, 32, 1024, 800)
attn_mask = torch.randn(1, 1, 1024, 1024)
