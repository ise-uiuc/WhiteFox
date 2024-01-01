
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        inv_scale = math.sqrt(query.size(-1))
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 8, 4)
key = torch.randn(1, 2, 16, 6)
value = torch.randn(1, 2, 16, 4)
