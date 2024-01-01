
class Model(torch.nn.Module):
    def __init__(self, query_channels, key_channels):
        super().__init__()
        self.query_channels = query_channels
        self.key_channels = key_channels
 
    def forward(self, query, key, value, dropout_p=0.0):
        q, k = query.unsqueeze(2), key.unsqueeze(1)
        w = torch.matmul(q, k).div(self.query_channels ** 0.25)
        w = torch.nn.functional.dropout(w, p=dropout_p)
        w = w.softmax(dim=-1)
        v = torch.matmul(w, value.unsqueeze(-1)).squeeze(-1)
        return v

# Initializing the model
m = Model(3, 4)

# Inputs to the model
query = torch.randn(1, 4, 8)
key = torch.randn(1, 8, 4)
value = torch.randn(1, 8, 5)
output = m(query, key, value)

