
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.head_size = 64
        self.all_head_size = self.num_heads * self.head_size
 
    def forward(self, q, k, v, attn_mask):
        qk = q @ k.transpose(-2, -1)
        qk = qk / math.sqrt(q.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = attn_weight + attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_output = attn_weight @ v
        return attn_output

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 196, 128)
x2 = torch.randn(1, 196, 128)
x3 = torch.randn(1, 196, 128)
x4 = torch.randn(1, 196, 128)
x5 = torch.randn(1, 1, 196)
x6 = torch.zeros(x2.size(0), 1, 196)
y = model(x1, x2, x3, x4) == model(x1, x2, x5, x6)

