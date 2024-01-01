
class SelfAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.tokeys  = nn.Linear(h, h, bias=False)
        self.toqueries = nn.Linear(h, h, bias=False)
        self.tovalues = nn.Linear(h, h)
 
    def forward(self, x):
        b, t, h = x.size()
 
        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)
 
        keys = keys.view(b, t, self.num_heads, h // self.num_heads).permute(0, 2, 1, 3)
        queries = queries.view(b, t, self.num_heads, h // self.num_heads).permute(0, 2, 1, 3)
        values = values.view(b, t, self.num_heads, h // self.num_heads).permute(0, 2, 1, 3)
 
        keys = keys. contiguous().view(b * self.num_heads, t, h // self.num_heads)
        queries = queries.contiguous().view(b * self.num_heads, t, h // self.num_heads)
        values = values.contiguous().view(b * self.num_heads, t, h // self.num_heads)
 
        inv_scale = 1 / (h // self.num_heads) ** 0.5
        scaled_dot_product = torch.matmul(queries, keys.transpose(-2, -1)) * inv_scale
        attention_weights = scaled_dot_product.softmax(-1)
        output = torch.matmul(attention_weights, values)
 
        output = output.view(b, self.num_heads, t, h // self.num_heads)
        output = output.permute(0, 2, 1, 3).contiguous().view(b, t, h)
 
        return output

# Initializing the model
# Dimensionality of the embedding space, number of attention heads
d, h = 32, 4
m = SelfAttention(h)

# Inputs to the model
x1 = torch.randn(1, 128, d)
