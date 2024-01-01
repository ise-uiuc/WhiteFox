
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 576
        self.dim = 384 // self.heads # Not sure how to get heads, but 2304 is some multiple of 1152
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) # Compute the attn_score as the dot product of inputs
        qk = qk + attn_mask # Add attn_mask as an extra dimension
        attn_weight = torch.softmax(qk, dim=-1) # Softmax over the dimension of scores
        attn_weight = torch.dropout(attn_weight, 0.0, True)
        output = attn_weight @ value # Compute attn_output as the dot product of softmaxed scores and value
        return output
# Inputs to the model
query = torch.randn(1, 1036, 256, 1208) # This should match the shape requirement of the attn_score as it's 2D.
key = torch.randn(1, 1036, 256, 1208) # Should have the shape of attn_weight and value
value = torch.randn(1, 1036, 256, 1208) # Should have the shape of attn_weight and key
attn_mask = torch.randn(1, 1, 256, 256)
