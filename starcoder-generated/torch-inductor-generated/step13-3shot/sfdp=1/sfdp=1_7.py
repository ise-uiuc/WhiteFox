
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, dropout_p=0.0, inv_scale_factor=1.0):
        query = query.float()
        key = key.float()
        value = value.float()
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, dropout_p)
        output = v4.matmul(value)
        return output 

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 32, 32)
value = torch.randn(1, 3, 32, 32)
