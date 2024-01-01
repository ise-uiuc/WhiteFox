
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(scale_factor)
        v3 = torch.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 768, 196)
key = torch.randn(1, 768, 512)
value = torch.randn(1, 768, 512)
scale_factor = torch.randn(1)
