
class SimpleAttention(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
 
    def attention(self, q, k, v, attn_mask):
        attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_x = attn_weight @ v
        return attn_x
 
    def forward(self, q, k, v, attn_mask):
        h = q.size(-1)
        attn_x = self.attention(q.view(-1, h, self.num_heads), k.view(-1, h, self.num_heads), v, attn_mask)
        return attn_x.view(-1, h * self.num_heads)
 
# Initializing the model
m = SimpleAttention(num_heads=8)

# Inputs to the model
q = torch.randn(1, 80, 512)
k = torch.randn(1, 80, 512)
v = torch.randn(1, 80, 512)
attn_mask = torch.ones(1, 80, 256)
