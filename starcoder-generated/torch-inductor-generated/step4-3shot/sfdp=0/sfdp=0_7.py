
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale = 1.0 / math.sqrt(1000)
 
    def forward(self, query, key, value):
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / self.inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 1000, 10)
key = torch.randn(16, 100, 1000)
value = torch.randn(16, 100, 1000)
