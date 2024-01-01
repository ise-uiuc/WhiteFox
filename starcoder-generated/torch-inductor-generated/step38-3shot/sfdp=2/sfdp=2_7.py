
class Model(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super().__init__()
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
hidden_dim = 16
num_heads = 2
inv_scale_factor = 1.0 / math.sqrt(hidden_dim)
dropout_p = 0.1
m = Model(num_heads, hidden_dim)

# Inputs to the model
query = torch.randn(1, hidden_dim * num_heads, 8, 64)
key = torch.randn(1, hidden_dim * num_heads, 8, 64)
value = torch.randn(1, hidden_dim * num_heads, 8, 64)
