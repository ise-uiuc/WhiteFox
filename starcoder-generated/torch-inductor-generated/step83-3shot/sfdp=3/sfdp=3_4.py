
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor)
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=dropout_p)
        v4 = torch.matmul(v3, value)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 512)
key = torch.randn(1, 50, 256)
value = torch.randn(1, 50, 256)
scale_factor = torch.randn(12, 50)
dropout_p = torch.tensor([0.565])
