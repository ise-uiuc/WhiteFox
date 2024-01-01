
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, num_heads, input_len, intermediate_dim):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.input_len = input_len
        self.intermediate_dim = intermediate_dim
    
        self.scale = query_dim**0.5
        self.query_proj_weight = torch.nn.Parameter(torch.randn(num_heads, query_dim, query_dim))
        self.key_proj_weight = torch.nn.Parameter(torch.randn(num_heads, key_dim, key_dim))
        self.value_proj_weight = torch.nn.Parameter(torch.randn(num_heads, input_len, key_dim))
        self.output_proj_weight = torch.nn.Parameter(torch.randn(num_heads, intermediate_dim, num_heads * key_dim))
    
        self.query_proj_bias = torch.nn.Parameter(torch.randn(num_heads, query_dim, 1))
        self.key_proj_bias = torch.nn.Parameter(torch.randn(num_heads, key_dim, 1))
        self.value_proj_bias = torch.nn.Parameter(torch.randn(num_heads, input_len, 1))
        self.output_proj_bias = torch.nn.Parameter(torch.randn(num_heads, intermediate_dim, 1))
    
    def forward(self, query, key, value):
        q, k, v = self.query_proj.forward(query), self.key_proj.forward(key), self.value_proj.forward(value)
        k = torch.transpose(k, 1, 2)
        scaled_dot_product = torch.matmul(q, k) * self.scale
        attention_weights = scaled_dot_product.softmax(dim=2)
        output = torch.matmul(attention_weights, v)
        output = torch.transpose(torch.reshape(output, (1, self.num_heads * self.key_dim)), 0, 1)
        output = self.output_proj.forward(output)
        return output

# Initializing the model
m = Model(query_dim=16, key_dim=32, num_heads=2, input_len=64, intermediate_dim=32)

# Inputs to the model
query, key, value = torch.randn(1, 4, 16), torch.randn(1, 8, 32), torch.randn(1, 8, 64)
output = m.forward(query, key, value)

# Inputs to the model
query, key, value = torch.randn(1, 4, 16), torch.randn(1, 8, 32), torch.randn(1, 8, 64)
