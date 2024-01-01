
class Model(torch.nn.Module):
    def __init__(self, n_heads=3):
        super().__init__()
        self.query = torch.nn.Linear(16, 16*n_heads)
        self.key = torch.nn.Linear(16, 16*n_heads)

    def forward(self, x1, x2, inv_scale_factor, dropout_p):
		# Unpack data
        x1 = self.query(x1)
        x2 = self.key(x2)

        x1 = x1.reshape(x1.shape[:-1] + (1, 1, -1))
        x2 = x2.reshape(x2.shape[:-1] + (1, 1, -1))

        # Compute the dot product of the query and the key
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)

        # Reshpe the output
        output = output.reshape(*output.shape[:-2], self.n_heads*16)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
inv_scale_factor = torch.rand(1)*0.1 + 1e-6
dropout_p = torch.rand(1)
