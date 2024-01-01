
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
 
    def forward(self, query, key, value):
        v0 = torch.matmul(query, key.transpose(-2, -1))
        v1 = torch.nn.functional.dropout(v0, p=dropout_p)
        v2 = torch.matmul(v1, value)
        return v2

# Initializing the model
dropout_p = 0.2
m = Model(dropout_p)

# Inputs to the model
x = torch.randn(1, 32, 64)
y = torch.randn(1, 32, 32)
z = torch.randn(1, 32, 24)
