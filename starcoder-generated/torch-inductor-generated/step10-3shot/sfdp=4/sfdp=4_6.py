
class Model(torch.nn.Module):
    def forward(query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
model = Model()

# Inputs to the model
query = torch.randn(20, 128, 50)
key = torch.randn(20, 128, 64)
value = torch.randn(20, 128, 64)
attn_mask = torch.randn((20, 1, 50, 64))
