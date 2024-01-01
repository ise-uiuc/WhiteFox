
class MultiHeadAttention(nn.Module):
    def __init__(self, query_channels, key_channels, output_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_channels = key_channels
        self.output_channels = output_channels
        self.q_lin = nn.Linear(query_channels, num_heads * key_channels)
        self.k_lin = nn.Linear(key_channels, num_heads * key_channels)
        self.v_lin = nn.Linear(key_channels, num_heads * output_channels)
        self.out_lin = nn.Linear(num_heads * output_channels, output_channels)
        self.dropout = nn.Dropout(0.25)

    def forward(self, query, key, value):
        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)
        q, k, v = (
            (
                x.view(x.size(0), x.size(1), self.num_heads, self.key_channels)
            ) for x in (q, k, v)
        )
        scaled_q, scaled_k, scaled_v = (
            (x.transpose(-2, -1)) for x in (q, k, v)
        )
        attn_logits = scaled_q.matmul(scaled_k)
        attn = safe_logsoftmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = attn.matmul(scaled_v)
        out = out.transpose(0, 1).contiguous().view(
            attn.size(1), -1)
        return self.out_lin(out)

    def extra_repr(self):
        return "MultiHeadAttention(%d, %d, %d)" % (
            self.key_channels, self.output_channels, self.num_heads
        )

# Initializing the model
m = MultiHeadAttention(query_channels=256, key_channels=256, output_channels=256, num_heads=4)

# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 1, 8, 8)
