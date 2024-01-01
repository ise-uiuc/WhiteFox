
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 128
        self.seq_len = 1
        self.dim = 1024 // self.heads
        self.query = torch.nn.Parameter(torch.randn(1, 128, self.seq_len, self.dim))
        self.key = torch.nn.Parameter(torch.randn(1, 128, self.seq_len, self.dim))
        self.value = torch.nn.Parameter(torch.randn(1, 128, self.seq_len, self.dim))
        self.attn_mask = torch.nn.Parameter(torch.randn(1, 1, self.seq_len, self.seq_len))
    def forward(self, query=self.query, key=self.key, value=self.value, attn_mask=self.attn_mask):
        # This is the same implementation from above, and only for illustrative purpose.
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.6, True)
        output = attn_weight @ value
        return output
