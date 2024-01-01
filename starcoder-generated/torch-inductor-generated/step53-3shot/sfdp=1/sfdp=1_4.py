
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.0):
        super().__init__()
        self.query_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.key_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.value_projection = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        a = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            mask = mask[:, None, None, :]
            a = a.masked_fill(mask == 0, -np.inf)
        # a: (batch_size, num_heads, query_len, key_len)
        inv_scale_factor = np.sqrt(k.shape[-1])
        scaled_a = a / inv_scale_factor
        scaled_a = F.softmax(scaled_a, dim=-1)
        a = scaled_a
        if dropout:
            a = F.dropout(scaled_a, p=dropout)
        # a: (batch_size, num_heads, query_len, key_len)
        output = torch.matmul(a, v)
        return output

# Initializing the model
m = Model(1024, num_heads=8)

# Inputs to the model. The "mask" input is optional, but we provide an example here. You should not use the mask input.
query = torch.randn(1, 128,  1024)
key   = torch.randn(1, 128, 16, 1024)
value = torch.randn(1, 128, 16, 1024)
mask  = torch.zeros(1, 128, 16).to(torch.bool)

