
class Model(torch.nn.Module):
    def __init__(self, query_len, key_len, value_len, hidden_size, scale_factor, dropout_p):
        super().__init__()
 
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
query_len = 500
key_len = 400
value_len = 400
hidden_size = 128
scale_factor = 1 / (hidden_size ** 0.5)
dropout_p = 0.5
m = Model(query_len, key_len, value_len, hidden_size, scale_factor, dropout_p)

# Inputs to the model
query = torch.randn(1, query_len, hidden_size)
key = torch.randn(1, key_len, hidden_size)
value = torch.randn(1, value_len, hidden_size)
dropout_p = 0.5
