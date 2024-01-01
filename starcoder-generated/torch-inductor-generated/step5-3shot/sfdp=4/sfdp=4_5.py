
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, attn_mask=None):
        # Compute the dot product of the query and key, and scale it
        qk = query @ key.transpose(-2, -1)
        dimension_scale = math.sqrt(query.size(-1))
        attn_weights = qk / dimension_scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0)
        # Apply softmax to the result
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # Compute the dot product of the attention weights and the value
        attn_output = attn_weights @ value
        return attn_output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
