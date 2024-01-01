
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0
        self.d_per_head = d_model // self.num_heads
        self.key_dim = key_dim

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(key_dim, d_model)
        self.value = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value, bias, attn_mask=None):
        residual = query

        q = self.query(query).view(
            residual.size()[0],
            residual.size()[1],
            self.num_heads,
            self.d_per_head)  # (bs, length, num_heads, d_per_head)
        k = self.key(key).view(
            residual.size()[0],
            residual.size()[1],
            self.num_heads,
            self.d_per_head)  # (bs, length, num_heads, d_per_head)
        v = self.value(value).view(
            residual.size()[0],
            residual.size()[1],
            self.num_heads,
            self.d_per_head)  # (bs, length, num_heads, d_per_head)

        q = q.permute(0, 2, 1, 3).contiguous().view(
            -1, residual.size()[1], self.d_per_head)  # (bs * num_heads, length, d_per_head)
        k = k.permute(0, 2, 1, 3).contiguous().view(
            -1, residual.size()[1], self.d_per_head)  # (bs * num_heads, length, d_per_head)
        v = v.permute(0, 2, 1, 3).contiguous().view(
            -1, residual.size()[1], self.d_per_head)  # (bs * num_heads, length, d_per_head)
        if bias is not None:
            attn_mask = bias.view(-1, 1, residual.size()[1], 1).repeat(
                1, self.num_heads, 1, 1)
            k = k.masked_fill(attn_mask, -1e10)
        qk = torch.bmm(q, k.transpose(1, 2))
        qk /= math.sqrt(self.d_per_head)
        if attn_mask is not None:
            qk += attn_mask

        attn_weight = nn.Softmax(dim=2)(qk)

        output = torch.bmm(attn_weight, v)
        output = output.view(residual.size()[0], self.num_heads,
                             residual.size()[1], self.d_per_head)
        output = output.permute(0, 2, 1, 3).contiguous().view(
            residual.size()[0], residual.size()[1], -1)
        output += residual

        return output

# Initializing the model
num_heads = 4
d_model = 256
key_dim = 16
m = MultiHeadAttention(num_heads, d_model, key_dim)

# Inputs to the model
query = torch.randn(4, 60, d_model)
key = torch.randn(4, 120, key_dim)
value = torch.randn(4, 120, d_model)
attn_mask = torch.randn(4, num_heads, 60, 120).gt(0)
bias = torch.randn(1, 1, 1, 120)
