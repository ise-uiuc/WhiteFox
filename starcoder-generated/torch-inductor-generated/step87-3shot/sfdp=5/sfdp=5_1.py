
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 2048
        self.dim = 1024 // self.heads
    def forward(self, query, key, value, src_mask, tgt_mask):
        q = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        q = q + src_mask
        k = key @ key.transpose(-2, -1) / math.sqrt(key.size(-1))
        k = k + tgt_mask
        attn_weight = torch.softmax(q, dim=-1) @ torch.softmax(k, dim=-1).transpose(-2, -1)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.5, True)
        output = attn_weight @ value
        return output
    def forward(self, query, key, value, src_mask, tgt_mask):
        q = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        q = q + src_mask
        k = key @ key.transpose(-2, -1) / math.sqrt(key.size(-1))
        k = k + tgt_mask
        attn_weight = torch.softmax(q, dim=-1) @ torch.softmax(k, dim=-1).transpose(-2, -1)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.5, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(64, 4, 2048 // 64, 1024)
key = torch.randn(64, 4, 2048 // 64, 1024)
value = torch.randn(64, 4, 2048 // 64, 1024)
src_mask = torch.randn(64, 1, 4, 4)
tgt_mask = torch.randn(64, 1, 4, 4)
