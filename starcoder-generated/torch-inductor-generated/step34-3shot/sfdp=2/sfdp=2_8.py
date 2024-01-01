
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_features):
        super().__init__()
        self.scale_factor = hidden_size ** -0.5
        self.dropout = torch.nn.Dropout(p=0.0)
        self.q = torch.nn.Linear(num_features, num_features, bias=False)
        self.k = torch.nn.Linear(num_features, num_features, bias=False)
        self.v = torch.nn.Linear(num_features, num_features, bias=False)
 
    def forward(self, query, key, value, dropout_p=0.1):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        max_dim = scaled_qk.shape[-1]
        dropout_qk = self.dropout(scaled_qk.softmax(dim=-1).to(torch.float64))
        dropout_qk = torch.nn.functional.dropout(dropout_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
hidden_size = 32
num_heads = 4
num_features = 128
m = Model(hidden_size, num_heads, num_features)

# Inputs of the original model
query = torch.randn(4, 20, hidden_size)
key = torch.randn(4, 40, hidden_size)
value = key
