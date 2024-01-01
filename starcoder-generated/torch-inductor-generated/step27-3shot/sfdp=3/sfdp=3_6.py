
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()

        self.d_model = d_model
        self.heads = heads

        self.scale = torch.sqrt(torch.FloatTensor([d_model])).item()

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        scale = self.scale

        batch_size = query.shape[0]
        N = value.shape[0]

        query = self.query_projection(query).view(batch_size, N, self.heads, self.d_model // self.heads).transpose(-2, -3)
        key = self.key_projection(key).view(batch_size, N, self.heads, self.d_model // self.heads).transpose(-2, -3)
        value = self.value_projection(value).view(batch_size, N, self.heads, self.d_model // self.heads).transpose(-2, -3)

        attn_weights = torch.matmul(query, key.transpose(-2,-1)) / scale

        if mask is not None:
            attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = attn_weights.softmax(dim=-1)

        context = torch.matmul(attn_weights, value)
        context = context.transpose(-2, -3)

        new_shape = [*context.shape[:-2], self.d_model]
        context = context.reshape(new_shape)

        return context

# Initializing the model
m = MultiHeadAttention(d_model=512, heads=8)

# Inputs to the model
query_tensor = torch.randn(4, 49, 512)
key_tensor = torch.randn(4, 49, 512)
value_tensor = torch.randn(4, 49, 512)
mask = torch.ones(4, 49, 49)

output = m(query_tensor, key_tensor, value_tensor, mask=mask)
