
class Model(torch.nn.Module):
    def __init__(self, query_dim, value_dim):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim

        self.scale_factor = query_dim ** -0.5

        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))

        scaled_qk = qk.mul(self.scale_factor)

        dropout_qk = self.dropout(scaled_qk.softmax(dim=-1))

        output = dropout_qk.matmul(value)

        return output

# Initializing the model
m = Model(10, 20)

# Inputs to the model
query = torch.randn(2, 5, 10)
key = torch.randn(2, 100, 10)
value = torch.randn(2, 100, 20)
