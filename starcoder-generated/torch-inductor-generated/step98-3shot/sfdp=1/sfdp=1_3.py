
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p=0.0, num_heads=8, scale_factor=None):
        super().__init__()
        self.scale_factor = dim ** -0.5 if scale_factor is None else scale_factor
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, values, keys, queries, mask=None):
        # Calculate the scaled dot product
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)

        # Apply the softmax
        softmax_qk = self.softmax(scaled_qk)

        # Apply dropout
        dropout_qk = self.dropout(softmax_qk)

        # Compute the dot product against the values
        output = dropout_qk.matmul(values)
        return output

# Initializing the model
dim = 1024
m = Model(dim, dropout_p=0.1)

# Inputs to the model
value = torch.rand(1, 8, 196, 1024)
keys = torch.rand(1, 8, 196, 1024)
queries = torch.rand(1, 8, 196, 1024)
