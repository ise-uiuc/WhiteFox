
class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_final = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask):
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        dim = query.dim()
        transpose_a = dim == 2
        transpose_b = dim == 1

        key = key.transpose(-2, -1)
        attn_weight = torch.bmm(query, key) / math.sqrt(query.size(-1))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask.expand(attn_weight.shape[0], -1, -1)
            attn_weight += attn_mask

        attn_weight = F.softmax(attn_weight, dim=-1)
        attn = torch.bmm(attn_weight, value)

        if transpose_a:
            attn = attn.transpose(1, 0)
        if transpose_b:
            attn = attn.transpose(1, 0)

        return self.linear_final(attn)

# Initializing the model
m = MultiHeadedAttention(10, 3)

# Inputs to the model
query = torch.randn(1, 4, 10)
key = torch.randn(1, 6, 10)
value = torch.randn(1, 6, 10)
attn_mask = torch.ones(1, 1, 1)

