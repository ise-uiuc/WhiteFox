
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        __x1__ = torch.matmul(query, key.transpose(-2, -1))
        v1 = __x1__.div(scale_factor)
        v2 = torch.nn.functional.dropout(v1.softmax(dim=-1), p=dropout_p)
        output = v2.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
num_samples = 2
d_model = 5
num_heads = 3

query = torch.randn(num_samples, d_model, num_heads, d_model//num_heads)
key = torch.randn(num_samples, d_model, num_heads, d_model//num_heads)
value = torch.randn(num_samples, d_model, num_heads, d_model//num_heads)
__scale_factor__ = 0.1
__dropout_p__ = 0.25
