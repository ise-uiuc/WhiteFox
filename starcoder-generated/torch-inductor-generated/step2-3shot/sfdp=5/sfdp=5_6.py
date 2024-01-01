
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
 
    def forward(self, query, key, value, attn_mask, dropout_p):
        qk = query @ key.transpose(-2, -1) / math.sqrt(self.num_heads)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model(2)

# Inputs to the model
query = torch.randn(1, 2, 8, 4)
key = torch.randn(1, 2, 8, 8)
value = torch.randn(1, 2, 8, 8)
attn_mask = torch.softmax(torch.randn(1, 2, 8, 8) * - 10000, dim=-1)
dropout_p = torch.tensor(0.5)
