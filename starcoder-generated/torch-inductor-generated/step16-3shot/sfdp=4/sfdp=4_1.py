
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_o = torch.nn.Linear(dim, dim)
  
    def forward(self, query, key, value, attn_mask):
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        q_ = q.reshape(*q.shape[:-1], self.num_heads, q.shape[-1] // self.num_heads).transpose(-3, -2)
        k_ = k.reshape(*k.shape[:-1], self.num_heads, k.shape[-1] // self.num_heads).transpose(-3, -2)
        v_ = v.reshape(*v.shape[:-1], self.num_heads, v.shape[-1] // self.num_heads).transpose(-3, -2)
        qk = q_ @ k_.transpose(-2, -1) / math.sqrt(v_.shape[-1]) 
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v_
        output = output.transpose(-3, -2).reshape(*output.shape[:-2], output.shape[-2], output.shape[-1] * self.num_heads)
        output = self.to_o(output)
        return output

# Initializing the model
m = Model(dim=64, num_heads=8)

# Inputs to the model
x1 = torch.randn(1, 10, 64)
x2 = torch.randn(1, 15, 64)
attn_mask = torch.randint(0, 2, size=(x1.shape[0], x1.shape[1], x2.shape[1])).type_as(x1)
