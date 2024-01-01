
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
        return output
    
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 512)
key = torch.randn(1, 64, 512)
value = torch.randn(1, 64, 512)
inv_scale_factor = 1. / 512 
dropout_p = 0.2
