
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        v = torch.matmul(query, key.transpose(-2, -1))
        v = qk.div(inv_scale_factor)
        v = v.softmax(dim=-1)
        v = torch.nn.functional.dropout(v, p=dropout_p)
        v = v.matmul(value)
        return v

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 4, 3, 1)
key = torch.randn(2, 4, 3, 2)
value = torch.randn(2, 4, 3, 2)
inv_scale_factor = torch.rand(1)
dropout_p = torch.rand(1)
