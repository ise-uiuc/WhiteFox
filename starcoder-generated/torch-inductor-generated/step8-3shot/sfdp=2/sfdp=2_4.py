
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, max_length=2000):
        super().__init__()
        self.query_matrix = torch.nn.Parameter(torch.randn(query_dim, key_dim))
        self.key_matrix = torch.nn.Parameter(torch.randn(key_dim, key_dim))
        self.value_matrix = torch.nn.Parameter(torch.randn(query_dim, value_dim))
 
    def forward(self, query, key, value, dropout_p=0.1):
        query_key = torch.matmul(query, self.key_matrix).transpose(-1, -2)
        inv_scale_factor = np.power(key.size(-1), -0.5)
        softmax_qk = torch.softmax(query_key * inv_scale_factor, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, self.value_matrix)
        return output

# Initializing the model
m = Model(query_dim=8, key_dim=2000, value_dim=2000)

# Inputs to the model
query, key, value = [torch.randn(2, 4, 8) for _ in range(3)]
