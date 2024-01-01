
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, queries, keys, values, dropout_p, inv_scale_factor):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(queries, keys.transpose(-2, -1))
 
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(inv_scale_factor)
 
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
 
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
 
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(values)
 
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 8, 3, 64, 64)
keys = torch.randn(1, 8, 3, 64, 64)
values = torch.randn(1, 8, 3, 64, 64)
dropout_p = 0.2
inv_scale_factor = torch.randn(1, 8, 1, 1).div_(1e-5)
