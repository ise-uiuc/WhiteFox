
class Model(torch.nn.Module):
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk =  torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
model = Model()

# Dimensions of the tensors needed for initializing the model
query_dim, key_dim = 32, 32
value_dim, scale_factor_dim = 64, 32
batch_size, num_heads, sequence_length = 1, 2, 4
dropout_p = 0

# Instantiating the model
model = Model(query_dim, key_dim, value_dim, scale_factor_dim, dropout_p)

# Inputs to the model
query, key = torch.randn(batch_size, num_heads, sequence_length, query_dim), torch.randn(batch_size, num_heads, query_dim, sequence_length)
value = torch.randn(batch_size, num_heads, query_dim, value_dim)
scale_factor = torch.full([batch_size], scale_factor_dim, dtype=torch.float)
