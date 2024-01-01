
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, mask, dropout_p=0.3):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        inv_scale_factor = 1.0 / np.sqrt(np.prod((key.shape[-1])))
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
 
        if (mask is not None):
            output = output.masked_fill(mask, -1e9)
 
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 100)
key = torch.randn(1, 24, 100)
value = torch.randn(1, 24, 100)
mask = torch.ones(1, 12, 24).bool()
