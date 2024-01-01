
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 51
        self.seq_len = 178
        self.dim = 64 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        # Add a transpose in the middle to change the default seq_length from 178 to 256
        output = output.transpose(0, 2)
        return output
# Inputs to the model
query = torch.randn(256, 51, 178, 64)
key = torch.randn(256, 51, 178, 64)
value = torch.randn(256, 51, 178, 64)
attn_mask = torch.randn(1, 1, 178, 178)
