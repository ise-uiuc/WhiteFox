
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 128
        self.seq_len = 48
        self.dim = 1488 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.33628640918691906, True)
        output = attn_weight @ value
        return output 