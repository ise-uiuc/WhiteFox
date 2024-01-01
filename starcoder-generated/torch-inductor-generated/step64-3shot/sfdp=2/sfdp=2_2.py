
class Model(torch.nn.Module):
    def __init__(self,
                 query,
                 key,
                 value,
                 scale_factor,
                 dropout_p):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
# Shape of the query tensor: (batch_size, heads, sequence_length, sequence_length)
query = torch.randn(1, 2, 4, 6)
# Shape of the key tensor: (batch_size, heads, sequence_length, sequence_length)
key = torch.randn(1, 2, 12, 4)
# Shape of the value tensor: (batch_size, heads, sequence_length, sequence_length)
value = torch.randn(1, 2, 16, 12)
# Shape of the inverse scale factor: (1)
inv_scale_factor = torch.tensor(0.2)
# Shape of the dropout probability: (1)
dropout_p = torch.tensor(0.5)
m = Model(query, key, value, inv_scale_factor, dropout_p)

# Inputs to the model
