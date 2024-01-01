
class Model(torch.nn.Module):
    def __init__(self, query_dim):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(query_dim, query_dim), requires_grad=True)

    def forward(self, key, value, mask = None):
        # The mask argument here is optional
        if mask is not None:
            mask = mask.unsqueeze(1)
        query = self.query.view(1, query_dim, query_dim)
        inv_scale = math.sqrt(query_dim)

        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale

        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output, attention_weights

# Initializing the model with the query dimension
m = Model(query_dim)

# Inputs to the model
key = torch.randn(1, 1, 1, query_dim)
value = torch.randn(1, 1, 1, query_dim)
__output__, __weights__ = m(key, value)

