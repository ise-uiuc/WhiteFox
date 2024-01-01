
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 8, 288, 288)
key = torch.randn(3, 8, 288, 288)
value = torch.randn(3, 8, 288, 288)
scale_factor = torch.tensor([[128]])
dropout_p = 0.3
