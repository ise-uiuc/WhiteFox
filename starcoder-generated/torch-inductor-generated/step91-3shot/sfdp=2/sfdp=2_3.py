
class Model(torch.nn.Module):
    def __init__(self, query_tensor, key_tensor):
        super().__init__()
        self.query_dim = query_tensor.shape[-2]
        self.key_dim = key_tensor.shape[-2]

        self.matmul_query_key = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        self.softmax_query_key = torch.nn.functional.softmax(self.matmul_query_key/math.sqrt(self.query_dim))
        self.dropout_query_key = torch.nn.functional.dropout(self.softmax_query_key)

    def forward(self, value_tensor):
        return self.dropout_query_key.matmul(value_tensor)

# Initializing the model
q = torch.randn(1, 3, 64)
k = torch.randn(2, 3, 64)
v = torch.randn(2, 3, 64)
m = Model(q, k)

# Inputs to the model
