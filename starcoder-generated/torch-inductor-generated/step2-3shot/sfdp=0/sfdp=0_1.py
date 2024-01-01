
class Model(torch.nn.Module):
    def forward(self, input_tensor):
        query = torch.rand(1, 200, 250, 100)
        key = torch.rand(1, 100, 450, 300)
        value = torch.rand(1, 100, 450, 300)
        inv_scale = 100
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output

# Initializing the model
m = Model()
print(m)

# Inputs to the model
input_tensor = torch.randn(1, 200, 250, 100)
