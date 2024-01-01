
class AttentionLayer(nn.Module):
    # Constructor
    def __init__(self, input_dim, num_heads, dropout_prob):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dense = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    # Forward pass
    def forward(self, query, key, value, attn_mask):
        # Compute the dot product of the query and key, and scale it
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        # Add the attention mask to the scaled dot product
        qk = qk + attn_mask
        # Apply softmax to the result
        attn_weight = nn.functional.softmax(qk, dim=-1)
        # Compute the dot product of the attention weights and the value
        output = attn_weight @ value
        return output

# Initializing the model
layer = AttentionLayer(input_dim=512, num_heads=8, dropout_prob=0.5)

# Inputs to the model
query = torch.randn(4, 8, 512)
key = torch.randn(4, 8, 512)
value = torch.randn(4, 8, 512)
attn_mask = torch.tensor([(-1e9), (-1e9)], dtype=torch.float32)
