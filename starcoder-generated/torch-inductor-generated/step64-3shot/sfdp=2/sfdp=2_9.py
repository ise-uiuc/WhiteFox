
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk * inv_scale_factor # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = torch.matmul(dropout_qk, v) # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
batch_size = 2
heads = 4
d_model = 64
n_queries = 128
n_keys = 128
n_values = 128
sequence_length = 1024

inv_scale = 1 / (d_model ** 0.5)
dropout_p = 0.0

q = torch.randn(batch_size * n_queries * heads, sequence_length, d_model)
k = torch.randn(batch_size * n_keys * heads, sequence_length, d_model)
v = torch.randn(batch_size * n_values * heads, sequence_length, d_model)
