
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 96
        self.num_heads = 16
        self.scaling = self.dim**-0.5
        self.query_projection = torch.nn.Linear(self.dim, self.dim)
        self.key_projection = torch.nn.Linear(self.dim, self.dim)
        self.value_projection = torch.nn.Linear(self.dim, self.dim)
 
    def forward(self, query, key, value):
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        inv_scale = self.scaling / query.shape[0]**0.5
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) * inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 96)
key = torch.randn(1, 1, 96)
value = torch.randn(1, 1, 96)
