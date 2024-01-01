
class Model(torch.nn.Module):
    def __init__(self, n_dim, n_hid, dropout_p):
        super().__init__()
        self.n_hid, self.dropout_p = n_hid, dropout_p

        self.query = torch.nn.Parameter(torch.randn(n_dim, n_hid)) # Initialize the query in the attention mechanism
        self.key = torch.nn.Parameter(torch.randn(n_dim, n_hid)) # Initialize the key in the attention mechanism
        self.value = torch.nn.Parameter(torch.randn(n_dim, n_hid)) # Initialize the value in the attention mechanism
        self.inv_scale_factor = torch.nn.Parameter(torch.tensor(1.0 / np.sqrt(n_hid))) # Inverse square root of a scaling factor

    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(0, 1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(self.value) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
dropout_p = 0.5
m = Model(64, 32, dropout_p)

# Inputs to the model
x1 = torch.randn(6, 64)
