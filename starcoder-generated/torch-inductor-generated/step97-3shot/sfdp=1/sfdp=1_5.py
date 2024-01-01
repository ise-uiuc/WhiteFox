
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V, inv_scale_factor, dropout_p):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(Q, K.transpose(-2, -1))
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(inv_scale_factor)
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(V)
        return output

# Initializing the model
m = Model()

# Query tensor
Q = torch.randn(1, 8, 128, 128)
# Key tensor
K = torch.randn(1, 8, 128, 128)
# Value tensor
V = torch.randn(1, 8, 128, 128)
# Inverse scale factor
inv_scale_factor = torch.Tensor([1.0])
# Dropout probability
dropout_p = 0.0
