
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.QKV = torch.nn.Linear(dim, dim * 3, bias=False)
 
    def forward(self, x1):
        qkv = self.QKV(x1)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q / math.sqrt(q.size(-1))
        qk = q @ k.transpose(-1, -2)
        attn_w = torch.softmax(qk, dim=-1) 
        return attn_w @ v
 
# Initializing the model
m = Model(32, 4)

# Input to the model
x1 = torch.randn(2, 4, 32)
