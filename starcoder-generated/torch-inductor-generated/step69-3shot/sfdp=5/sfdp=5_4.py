
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.seq_len = 327
        self.dim = 64 // self.heads
    def forward(self, query, key, value, attn_mask):
        output = torch.randn(1, 256, 327, 64)
        for x in range(256):
            qk = query[:, x:, :, :] @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
            qk = qk[:x:] + attn_mask
            attn_weight = torch.softmax(qk, dim=-1)
            attn_weight = torch.dropout(attn_weight, 0.1, True)
            output[:, x:, :, :] = output[:, x:, :, :] + attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 256, 327, 64)
key = torch.randn(1, 256, 327, 64)
value = torch.randn(1, 256, 327, 64)
attn_mask = torch.randn(1, 255, 327, 327)
