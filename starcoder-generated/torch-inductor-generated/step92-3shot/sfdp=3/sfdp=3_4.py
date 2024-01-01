
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, queries, keys, values, scale_factor, dropout_p):
        dot_product = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_dot_product = dot_product.mul(scale_factor)
        softmax_output = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        dropout_softmax_output = torch.nn.functional.dropout(softmax_output, p=dropout_p)
        output = dropout_softmax_output.matmul(values)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 3, 64, 64)
keys = torch.randn(1, 3, 64, 64)
values = torch.randn(1, 3, 64, 64)
scale_factor = torch.tensor(1.0/math.sqrt(64))
dropout_p = 0.0
