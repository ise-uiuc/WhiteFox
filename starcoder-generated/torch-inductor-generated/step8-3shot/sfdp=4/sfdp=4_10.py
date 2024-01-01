
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = int(self.num_heads * 32)
        self.w_qkv = torch.nn.Linear(8, self.num_heads * self.head_dim * 3)
 
    def forward(self, x, attn_mask):
        batch_size = x.size(0)
        k = v = x
        k = k.reshape([batch_size, -1, self.num_heads, self.head_dim])
        v = v.reshape([batch_size, -1, self.num_heads, self.head_dim])
        q, k, v = [x.transpose(1, 2).reshape([batch_size * self.num_heads, -1, self.head_dim]) for x in (q, k, v)]
        kq = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        attn_mask = attn_mask.repeat([1, self.num_heads])
        attn_weight = torch.softmax(kq + attn_mask, dim=-1)
        w_qkv = self.w_qkv(x)
        output = attn_weight @ v
        output = output.transpose(1, 2).reshape([batch_size, -1, self.num_heads * self.head_dim])
        output = output + w_qkv
        output = torch.relu(output)
        return output

# Initializing the model
m = Model(8)

# Inputs to the model
x = torch.randn(1, 8, 128)
attn_mask = torch.zeros([1, 128, 128])
