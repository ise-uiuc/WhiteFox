
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, num_value):
        super().__init__()
        self.query_proj = torch.nn.Sequential(torch.nn.Linear(query_dim, key_dim))
        self.key_proj = torch.nn.Sequential(torch.nn.Linear(query_dim, key_dim))
        self.inv_scalar = torch.div(1, math.sqrt(query_dim))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.value_proj = torch.nn.Sequential(torch.nn.Linear(query_dim, query_dim * num_value))
 
    def forward(self, query, key, value):
        q = self.query_proj(query)
        k = self.key_proj(key)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * self.inv_scalar
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return torch.matmul(dropout_qk, self.value_proj(value).view(query.size(0), query.size(1), -1))
 
 
# Initializing the model
m = Model(8, 4, 10)

# Inputs to the model
query = torch.randn(1, 8)
key = torch.randn(10, 4)
value = torch.randn(10, 8)
