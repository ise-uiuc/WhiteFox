
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, scale_factor, dropout_p):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul1 = torch.nn.Linear(query_dim, key_dim)
        self.matmul2 = torch.nn.Linear(key_dim, value_dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)
    def forward(self, query, key, value, dropout_p):
        query = self.matmul1(query)
        key = self.matmul1(key)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, self.matmul2(value))
        return output

# Initializing the model
query_dim, key_dim, value_dim, scale_factor, dropout_p = 3, 3, 3, 0.2, 0.5
m = Model(query_dim, key_dim, value_dim, scale_factor, dropout_p)

# Inputs to the model
query = torch.randn(5, 7, query_dim)
key = torch.randn(5, 10, key_dim)
value = torch.randn(5, 10, value_dim)
