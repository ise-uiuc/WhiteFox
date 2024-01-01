
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        Q = query
        K = key
        V = value
        A = Q @ K.T
        inv_scale = 1.0 / math.sqrt(d_k)
        A = A * inv_scale
        A = torch.softmax(A, dim=-1)
        B = A @ V
        output = torch.nn.functional.dropout(B, p=dropout, training=self.training)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 3, 64)
key = torch.randn(2, 3, 64)
value = torch.randn(2, 3, 64)
