
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.query = torch.nn.Linear(query_dim, key_dim)
        self.key = torch.nn.Linear(key_dim, key_dim)
        self.value = torch.nn.Linear(value_dim, value_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.tanh = torch.nn.Tanh()
 
    def forward(self, query, key, value, dropout_p, inv_scale_factor=None):
        if inv_scale_factor is None:
            inv_scale_factor = torch.tensor(1.0) / math.sqrt(query_dim)
 
        output = self.tanh(self.dropout(torch.matmul(self.query(query), self.key(key).transpose(-2, -1)) / inv_scale_factor))
        output = output.matmul(self.value(value))
        return output

# Initializing the model
m = Model(query_dim=512, key_dim=512, value_dim=512)

# Inputs to the model
query = torch.randn(1, 64, 512)
key = torch.randn(1, 512, 128)
value = torch.randn(1, 512, 128)
dropout_p = 0.0
inv_scale_factor = None
