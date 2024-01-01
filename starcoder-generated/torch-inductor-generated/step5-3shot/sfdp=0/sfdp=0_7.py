
class Model(torch.nn.Module):
    def __init__(self, dim, inv_scale):
        super().__init__()
        self.inv_scale = inv_scale
 
    def forward(self, query, key, value):
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / self.inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output

# Initializing the model
m = Model(dim, inv_scale=1 / (dim ** 0.5))

# Inputs to the model
query = torch.randn(4, 3, dim)
key = torch.randn(4, 5, dim)
value = torch.randn(4, 5, dim)
