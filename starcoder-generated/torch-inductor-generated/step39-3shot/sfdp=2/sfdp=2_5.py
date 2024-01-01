
class Model(torch.nn.Module):
    def __init__(self, num_heads, query_dim, key_dim, dropout_p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = torch.nn.Linear(query_dim, num_heads * key_dim)
        self.key_proj = torch.nn.Linear(key_dim, num_heads * query_dim)
        self.value_proj = torch.nn.Linear(key_dim, num_heads * key_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.scaling_factor = math.sqrt(key_dim // num_heads) 

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scaling_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
num_heads = 2
query_dim = 2
key_dim = 2
dropout_p = 0.0

m = Model(num_heads, query_dim, key_dim, dropout_p)

# Inputs to the model
query = torch.randn(query_dim, num_heads, 4)
key = torch.randn(key_dim, num_heads, 7)
value = torch.randn(key_dim, num_heads, 6)
