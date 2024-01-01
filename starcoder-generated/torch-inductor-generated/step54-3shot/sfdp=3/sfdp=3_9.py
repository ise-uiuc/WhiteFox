
class Model(torch.nn.Module):
    def __init__(self, query_dim=12, key_dim=13, value_dim=14, scale_factor: float=0.0001, dropout_p=1e-6):
        super().__init__()
        self.query_softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.value_transform = torch.nn.Linear(value_dim, query_dim, False)

    def forward(self, query_tensor, key_tensor, value_tensor):
        v1 = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = self.query_softmax(v2)
        v4 = self.dropout(v3)
        v5 = self.value_transform(value_tensor)
        v6 = v4.matmul(v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
query_tensor = torch.randn(1, 40, 12)
key_tensor = torch.randn(1, 40, 13)
value_tensor = torch.randn(1, 40, 14)
