
class Model(torch.nn.Module):
    def __init__(self, batch_size=2, num_heads=2, query_len=8, key_len=8, channel=2):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(batch_size, num_heads, query_len, channel))
        self.key = torch.nn.Parameter(torch.randn(batch_size, num_heads, key_len, channel))
        self.value = torch.nn.Parameter(torch.randn(batch_size, num_heads, query_len, channel))
 
    def forward(self, qk, attn_mask):
        qk = qk @ self.key.transpose(-2, -1) / math.sqrt(self.query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.value
        return output

# Initializing the model
m = Model()

# Inputs to the model
qk = torch.randn(2, 2, 8, 2)
attn_mask = torch.randn(2, 2, 8)
