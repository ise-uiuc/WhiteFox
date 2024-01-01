
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = torch.nn.Linear(embedding + positional, hidden * 3)
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = (query.size(-2) * query.size(-1)) ** -.5 # Compute the inverse of the scale factor used to scale the dot product of the query and the key
        scaled_qk = qk.div(inv_scale_factor)
        a1 = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        output = torch.nn.functional.dropout(a1, p=dropout_p) # Apply dropout to the softmax output
        output = output.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(batch, 1, hidden_dimension)
key = torch.randn(batch, seq_len, hidden_dimension)
value = torch.randn(batch, seq_len, hidden_dimension)
dropout_p = 0.5
