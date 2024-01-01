
class Model(torch.nn.Module):
    def __init__(self, dim=192, num_heads=6, dropout_p=0.1):
        super().__init__()
        self.scale_factor = dim ** -0.5
        self.matmul_query = torch.nn.Linear(dim, dim)
        self.matmul_key = torch.nn.Linear(dim, dim)
        self.matmul_value = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.activation = torch.nn.functional.softmax
        
    def forward(self, query, key, value):
        q = self.matmul_query(query)
        k = self.matmul_key(key)
        v = self.matmul_value(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = self.activation(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 192)
key = torch.randn(1, 6, 192)
value = torch.randn(1, 6, 192)
