
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key tensors
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 5, 128)
key = torch.randn(1, 6, 128)
value = torch.randn(1, 6, 512)
inv_scale_factor = torch.randn(1)

