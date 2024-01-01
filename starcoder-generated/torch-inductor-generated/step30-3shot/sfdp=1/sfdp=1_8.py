
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        v6 = torch.matmul(query, key.transpose(-2, -1))
        v7 = v6.div(inv_scale_factor)
        v8 = v7.softmax(dim=-1)
        v9 = torch.nn.functional.dropout(v8, p=dropout_p)
        v10 = v9.matmul(value)
        return v10

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 1024, 64)
key = torch.randn(3, 1024, 64)
value = torch.randn(3, 1024, 64)
