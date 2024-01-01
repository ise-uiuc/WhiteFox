
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "Embedding dimension must be divisible by number of heads."
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, query, key, value):
        q, k, v = self.in_proj_qkv(query, key, value)
        q = q*self.scaling
        q = (self.split_heads(q)).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = nn.functional.softmax(
            attn, dim=-1)
        attn = nn.functional.dropout(attn, p=self.dropout)
        output = (attn @ v).transpose(1, 2).contiguous()
        output = self.merge_heads(output)
        return self.out_proj(output)

    def in_proj_qkv(self, query, key, value):
        return F.linear(query, self.in_proj_weight, bias=None), \
               F.linear(key, self.in_proj_weight, bias=None), \
               F.linear(value, self.in_proj_weight, bias=None)

    def split_heads(self, x, kv=False):
        if kv==False:
            return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).permute(2, 0, 1, 3)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), self.embed_dim)
        return x

# Initializing the model
m = MultiHeadAttention(embed_dim=16, num_heads=8)

# Inputs to the model
query = torch.randn(4, 10, 16)
key = torch.randn(4, 8, 16)
value = torch.randn(4, 8, 16)
