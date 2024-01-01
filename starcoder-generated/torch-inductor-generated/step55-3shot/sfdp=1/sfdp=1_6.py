
class Model(torch.nn.Module):
    def forward(self, q, k, v, scale_factor, dropout_p):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(q, k.transpose(-2, -1))
        # Scale the dot product
        scaled_qk = qk.div(inv_scale_factor)
        # Apply softmax to the data
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        # Compute the dot product of the dropout output and the value tensors
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
# Inputs to the model
q = torch.randn(1, 50, 16)
k = torch.randn(1, 100, 16)
v = torch.randn(1, 100, 16)
scale_factor = torch.tensor(1e4)
dropout_p = torch.tensor(0.5)
