
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout_p):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_project = torch.nn.Linear(query_dim, key_dim, bias=False)
        self.key_project = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.value_project = torch.nn.Linear(value_dim, value_dim, bias=False)
        self.dropout_p = dropout_p
   
    def forward(self, query, key, value):
        query = self.query_project(query)
        key = self.key_project(key)
        value = self.value_project(value)
        inv_scale_factor = math.sqrt(query.size(-1) * self.key_dim)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(query_dim=64, key_dim=64, value_dim=32, dropout_p=0.2)

# Inputs to the model (only showing one batch of two samples, not the full batch of 32 samples)
X1 = torch.randn(2, 64)
X2 = torch.randn(2, 64)
X3 = torch.randn(2, 32)
