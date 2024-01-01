
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.proj_query = nn.Linear(self.embed_dim, embed_dim)
        self.proj_key = nn.Linear(self.embed_dim, embed_dim)
        self.proj_value = nn.Linear(self.embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, attn_mask=None):
        query = self.proj_query(query)
        key = self.proj_key(key)
        value = self.proj_value(value)

        b, q_length, _ = query.size()
        _, k_length, _ = key.size()
        v_length = value.size(1)

        # check memory
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        assert qkv_same or kv_same, "All operands should share the same underlying data."

        q = query.contiguous().view(b, q_length, self.num_heads, self.head_dim)
        k = key.contiguous().view(b, k_length, self.num_heads, self.head_dim)
        v = value.contiguous().view(b, v_length, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3).contiguous().view(-1, q_length, self.head_dim)
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, k_length, self.head_dim)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, v_length, self.head_dim)

        q = q.repeat(self.num_heads, 1, 1)
        k = k.repeat(self.num_heads, 1, 1)
        v = v.repeat(self.num_heads, 1, 1)

        dots = torch.bmm(q, k.transpose(1, 2))
        scaled_dots = dots / np.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
            scaled_dots.masked_fill_(attn_mask, float('-inf'))
        attn = self.softmax(scaled_dots)
        if attn_mask is not None:
            attn.masked_fill_(attn_mask, 0)
        out = torch.bmm(attn, v)

        out = out.view(b, self.num_heads, q_length, self.head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, q_length, -1)
        return out

# Initializing the model
n = MultiheadAttention(embed_dim=64, num_heads=4)

# Inputs to the model
query = torch.randn(1, 24, 64)
key = torch.randn(1, 36, 64)
value = torch.randn(1, 36, 64)
mask = torch.randn(36, 24)
