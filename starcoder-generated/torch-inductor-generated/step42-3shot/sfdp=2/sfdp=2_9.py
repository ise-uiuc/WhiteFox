
class Model(torch.nn.modules.Module):
    def __init__(self, dim, dropout_p, inv_scale_factor):
        super().__init__()
        self.scale_factor = dim ** -0.5 # Compute the scale factor from the dimension
        self.dropout_p = dropout_p # Initialize the dropout value
        self.inv_scale_factor = inv_scale_factor # Initialize the inverse scale factor

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        dropped_qk = torch.nn.functional.dropout(scaled_qk, p=self.dropout_p) # Apply dropout to the scaled dot product
        output = dropped_qk.matmul(value) # Compute the dot product of the dropped output and the value
        return output

# Initializing the model
m = Model(dim=128, dropout_p=0.2, inv_scale_factor=1/(dim**0.5))

# Inputs to the model
query = torch.randn(33, 128) # Generate 33 inputs of dimension 128
key = torch.randn(32, 128) # Generate 32 inputs of dimension 128
value = torch.randn(32, 128) # Generate 32 inputs of dimension 128
