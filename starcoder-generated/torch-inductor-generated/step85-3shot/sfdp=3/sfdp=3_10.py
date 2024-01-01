
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):

        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))

        # Scale the dot product by a factor
        scaled_qk = qk.mul(1.0 / np.sqrt(np.sqrt(query.shape[-1])))

        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)

        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)

        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)

        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 32, 64)
value = torch.randn(1, 3, 32, 64)
