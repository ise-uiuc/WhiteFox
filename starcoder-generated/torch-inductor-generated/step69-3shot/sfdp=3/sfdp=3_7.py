
class Model(torch.nn.Module):
    def __init__(self, dim, heads, dim_head, dropout, input_shape):
        super().__init__()
        num_heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads
        self.heads = num_heads
        self.dropout = dropout

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = torch.nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = dots.softmax(dim=-1)
        attn = torch.nn.functional.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)       

# Initializing the model
dim = 3
heads = 2
dim_head= 4
dropout = 0.05
model = Model(dim, heads, dim_head, dropout)

# Inputs to the model
x = torch.randn(1, 1, 1, 3)
