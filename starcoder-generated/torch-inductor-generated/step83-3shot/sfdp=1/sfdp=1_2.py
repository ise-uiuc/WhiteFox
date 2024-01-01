
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = v4.matmul(value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 3, 3)
key = torch.randn(1, 16, 4, 4)
value = torch.randn(1, 16, 3, 3)
inv_scale_factor = 0.00205078125
dropout_p = 0.10000038146972656
