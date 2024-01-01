
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, inputs):
        query, key, value = inputs
        inv_scale = math.sqrt(key.shape[-1])
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
_inputs = [torch.randn(1, 4, 100), torch.randn(1, 4, 150), torch.randn(1, 4, 150)]
