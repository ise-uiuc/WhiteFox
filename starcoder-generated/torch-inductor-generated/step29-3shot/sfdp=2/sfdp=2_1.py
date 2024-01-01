
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, queries, keys, values):
        dot_product = torch.matmul(queries, keys.transpose(-2, -1)) # Compute the dot product
        scaled_dot_prod = dot_product / math.sqrt(query.size(-1)) # Scale the dot product by the square root of the query dimension
        softmax_dot_prod = F.softmax(scaled_dot_prod, dim=-1) # Apply softmax to the scaled dot product
        dropout_dot_prod = F.dropout(softmax_dot_prod, p=0.5, training=self.training) # Apply dropout p=0.5 (with training flag)
#         dropout_dot_prod = torch.nn.functional.dropout(softmax_dot_prod, p=0.5)
        output = torch.matmul(dropout_dot_prod, value) # Compute the dot product
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 64, 32)
