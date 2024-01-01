
class Model(torch.nn.Module):
    def __init__(self, num_heads = 8, d_model = 26, dropout_rate = 0.):
        super().__init__()
        self.attn_qk = torch.nn.Linear(d_model, num_heads)
        self.v = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
 
    def forward(self, query, key, value):
        # Concatenate the query and key tensors
        qk = self.attn_qk(query) * self.attn_qk(key)

        # Scale the dot product by a factor
        scale_factor = torch.sqrt(torch.Tensor([query.shape[-1]]))
        scaled_qk = qk.mul(scale_factor)

        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)

        # Apply dropout to the softmax output
        dropout_qk = self.dropout(softmax_qk)

        # Compute the dot product of the dropout output and the value tensor
        output = self.v(dropout_qk.matmul(value))
        return output

# Initializing the model
m = Model(num_heads = 2, d_model = 4, dropout_rate = 0.)

# Inputs to the model
query = torch.randn(2, 4)
key = torch.randn(2, 4)
value = torch.randn(2, 4)
