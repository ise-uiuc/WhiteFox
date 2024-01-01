
class Model(torch.nn.Module):
    def __init__(self, dropout_p, inv_scale_factor):
        super().__init__()
 
    def forward(self, query, key, value):
        v0 = torch.matmul(query, torch.transpose(key, 1, 2))
        v1 = v0.div(self.inv_scale_factor)
        v2 = torch.nn.functional.softmax(v1, dim=0)
        v3 = torch.nn.functional.dropout(v2, p=self.dropout_p)
        v4 = torch.matmul(value, v3)
        return v4

# Initializing the model
m = Model(dropout_p, inv_scale_factor)

# Inputs to the model
query = torch.randn(1, 16, 8)
key = torch.randn(1, 20, 8)
value = torch.randn(1, 20, 8)
