
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(query_size), requires_grad=True)
        self.key = torch.nn.Parameter(torch.rand(key_size), requires_grad=True)
        self.inv_scale_factor = math.sqrt(key_size)
 
    def forward(self, value, dropout_p):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk / self.inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
query_size = 256
key_size = 64
dropout_p = 0.1
m = Model(query_size, key_size)

# Initializing value with the shape of (batch_size, seq_len, value_size), where seq_len should be greater than 1 and query_size should be greater than key_size
value = torch.randn(1, 32, key_size * 2)

# Forward computation
dropout_p = 0.1
