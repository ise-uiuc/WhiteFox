
class Model(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value
        self.dropout_p = dropout_p
 
    def forward(self, x1):
        v1 = torch.matmul(self.query, self.key.transpose(-2, -1))
        v2 = v1 * x1
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        output = torch.matmul(v4, self.value)
        return v4

# Initializing the model
query = torch.randn(64, 64, 128)
key = torch.randn(64, 128, 128)
value = torch.randn(64, 128, 128)
dropout_p = 0.5
m = Model(query, key, value, dropout_p)

# Inputs to the model
x1 = torch.randn(1, 64, 128)
