
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 1024
        self.dim = 1024 // self.heads
    def forward(self, query, key, value, attn_mask):
        # Compute the attention weights by performing the dot product of the query and keys
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        # Add the attention mask
        qk = qk + attn_mask
        # Apply softmax to the scaled dot product of the query and keys
        attn_weight = torch.softmax(qk, dim=-1)
        # Apply dropout to the softmax results
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        # Perform the dot product of the attention weights and the values
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 256, 1024, 1024)
key = torch.randn(1, 256, 1024, 1024)
value = torch.randn(1, 256, 1024, 1024)
attn_mask = torch.randn(1, 1, 1024, 1024)
