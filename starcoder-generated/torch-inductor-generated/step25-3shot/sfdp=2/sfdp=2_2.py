
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
query = torch.randn(1, 8, 137, 20)
key = torch.randn(1, 8, 56, 34)
value = torch.randn(1, 8, 56, 34)
inv_scale_factor = torch.rand(1)
dropout_p = torch.rand(1)
