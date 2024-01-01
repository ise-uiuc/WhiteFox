
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = None
        self.dropout_p = None

    # Define the forward function
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.mul(self.scale_factor) # Scale the dot product by a factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 14, 14)
key = torch.randn(1, 8, 14, 14)
value = torch.randn(1, 8, 14, 14)
