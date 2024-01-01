
class Model(torch.nn.Module):
    def __init__(self, hidden_size: int, dropout_p: float):
        super().__init__()
 
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax()
        self.matmul_for_attention = torch.nn.Linear(
            in_features=2 * hidden_size, out_features=1)
        self.matmul_for_dropout = torch.nn.Linear(
            in_features=hidden_size, out_features=1)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = self.matmul_for_attention(query)
        softmax_qk = self.softmax(scaled_qk.mul(scale_factor.transpose(-2, -1)))
        dropout_qk = self.dropout(self.matmul_for_dropout(softmax_qk))
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
hidden_size = 32
dropout_p = 0.2
m = Model(hidden_size, dropout_p)

# Inputs to the model
query = torch.randn(16, 2, hidden_size)
key = torch.randn(16, 2, hidden_size)
value = torch.randn(16, 2, hidden_size)
output = m(query, key, value)
output

