
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
 
        self.query_linear = torch.nn.Linear(query_size, key_size)
        self.key_linear = torch.nn.Linear(key_size, key_size)
        self.value_linear = torch.nn.Linear(value_size, value_size)
 
        self.scale_factor = 1.0 / (query_size ** 0.5)
 
    def forward(self, x1, x2, x3):
        v1 = self.query_linear(x1)
        v2 = self.key_linear(x2)
        v3 = self.value_linear(x3)
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4 * self.scale_factor
        v6 = torch.nn.functional.softmax(v5.div(self.dropout_p), dim=-1)
        dropout_qk = torch.nn.functional.dropout(v6, p=self.dropout_p)
        v7 = torch.matmul(dropout_qk, v3)
        return v7

# Initializing the model
m = Model(query_size=16, key_size=8, value_size=4)

# Inputs to the model
query = torch.randn(3, 5, 16)
key = torch.randn(3, 4, 8)
value = torch.randn(3, 4, 4)
x1, x2, x3 = query, key, value
