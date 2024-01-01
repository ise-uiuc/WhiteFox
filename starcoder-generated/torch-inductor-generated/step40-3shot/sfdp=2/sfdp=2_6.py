
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, n_heads, query_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(query_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Query, key, value, and related parameters
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 61)
value = torch.randn(1, 8, 64, 61)
n_heads = 32
query_scale_factor = sqrt(1 / n_heads)
dropout_p = 0.1

# Calling the model
