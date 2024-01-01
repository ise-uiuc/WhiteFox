
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(query, key, value, dropout_p=0.5):
        key = torch.transpose(key, dim0=-2, dim1=-1) # Transpose the dimension of the key
        dot_product = torch.matmul(query, key) # Compute the dot product of the query and the key
        scale_factor = torch.sqrt(torch.tensor(query.size(-1))) # Set the scale factor to the square root of the dimension size of the query
        inv_scale_factor = 1.0 / scale_factor # Inverse the scale factor
        scaled_dot_product = dot_product * inv_scale_factor #  Scale the dot product by the inverse scale factor
        softmax_dot_product = torch.nn.functional.softmax(scaled_dot_product, dim=scaled_dot_product.dim() - 1) # Apply softmax to the scaled dot product
        final_dot_product = torch.nn.functional.dropout(softmax_dot_product, p=dropout_p) # Apply dropout to the softmax output
        result = final_dot_product.matmul(value) # Compute the dot product of the dropout output and the value
        return result

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(25, 1, 16)
key = torch.randn(25, 16, 2)
value = torch.randn(25, 2, 16)

