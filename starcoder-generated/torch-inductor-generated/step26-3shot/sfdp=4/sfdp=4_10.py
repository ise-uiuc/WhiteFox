
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
torch.manual_seed(42)
m = Model()

# Inputs to the model
query = torch.randn(4, 5, 6)
key = torch.randn(4, 7, 6)
value = torch.randn(4, 7, 10)
attn_mask = torch.zeros((4, 5, 7)) # (batch_size, num_heads, seq_len)
