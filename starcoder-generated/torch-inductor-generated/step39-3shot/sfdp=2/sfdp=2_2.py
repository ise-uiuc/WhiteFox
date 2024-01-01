
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p

    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1)) # Compute the dot product of two inputs
        inv_scale_factor = math.sqrt(self.input_dim)
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and a value
        return output

# Initializing the model
model = Model(input_dim, output_dim, dropout_p)

# Inputs 1 to the model
input_tensor_1 = torch.randn(batch_size, seq_len, input_dim)

# Inputs 2 to the model
input_tensor_2 = torch.randn(batch_size, seq_len, output_dim)

