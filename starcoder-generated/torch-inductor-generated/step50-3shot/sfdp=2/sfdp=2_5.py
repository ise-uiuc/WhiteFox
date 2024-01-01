
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout_p=0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        weight = torch.Tensor(query_dim, key_dim)
        self.query_key_score = torch.nn.Parameter(torch.nn.init.uniform_(weight, -0.001, 0.001))
 
    def forward(self, query, key, value):
        inv_scale_factor = torch.sqrt(query.size(-1)) + torch.sqrt(key.size(-1) - query.size(-1))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
 
        return output

# Initializing the model
m = Model(16, 64, 64)

# Inputs to the model
query = torch.randn(1, 4, 16)
key = torch.randn(1, 4, 64)
value = torch.randn(1, 4, 64)
