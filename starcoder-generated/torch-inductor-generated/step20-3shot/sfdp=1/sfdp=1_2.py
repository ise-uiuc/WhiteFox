
class Model(torch.nn.Module):
    def __init__(self, num_heads, batch_size, dropout_p):
        super().__init__()
        self.batch_size = batch_size
        self.dropout_p = dropout_p

    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model(1, 1, 0.5)

# Inputs to the model
query = torch.randn(1, 1, 12, 64)
key = torch.randn(1, 1, 12, 64)
value = torch.randn(1, 1, 12, 64)
inv_scale_factor = 1.0 / math.sqrt(64)
