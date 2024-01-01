
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        # Replace the values in the mask with large negative numbers.
        attn_mask = qk.new_ones(qk.size())
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask = 1.0 / attn_mask
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -10000.0)
        return qk * attn_mask, attn_mask

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 2, 3, 4)
k = torch.randn(1, 2, 3, 4)
v = torch.randn(1, 2, 5, 4)
__output__, __attn_mask__ = m(q, k, v)

