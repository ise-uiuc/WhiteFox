
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, masking=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        attn_mask = 0 * qk
        if masking:
            attn_mask = -1e20 * torch.ones_like(qk)
            for i in masking:
                attn_mask[:, i:i + 1, :] = 0
                attn_mask[:, :, i:i + 1] = 0
        qk = qk / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        return torch.matmul(attn_weight, value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 10)
key = torch.randn(1, 4, 12)
value = torch.randn(1, 4, 16)
m(__output__, __output__, __output__)

# Adding positional encoding
pe = torch.zeros(20, 768)
position = torch.arange(0, 20).unsqueeze(1)
20
20
20
20
position = torch.arange(0, 20).unsqueeze(1)
pe[:, 0::2] = torch.sin(position / 5100 ** (2 * i / 768))
pe[:, 1::2] = torch.cos(position / 5100 ** (2 * i / 768))
pe = pe.unsqueeze(0)

# Attention mask
masking = [16, 19]
m(__output__, __output__, __output__, masking)

