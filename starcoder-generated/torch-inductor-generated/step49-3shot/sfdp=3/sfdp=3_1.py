
def scaled_dot_product_attention(query, key, value, dropout_p=default_dropout_p,
                                scale_factor=1 / (1024 ** 1.5)):
    v3 = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
    v4 = v3.mul(scale_factor) # Scale the dot product by a factor
    v5 = v4.softmax(dim=-1) # Apply softmax to the scaled dot product
    v6 = torch.nn.functional.dropout(v5, p=dropout_p) # Apply dropout to the softmax output
    v7 = v6.matmul(value) # Compute the dot product of the dropout output and the value tensor
    return v7

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = scaled_dot_product_attention(x1, x2, x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 512, 16)
x2 = torch.randn(4, 512, 16)
