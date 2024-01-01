
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1/(query.shape[-1]**0.25)

        v1 = qk * inv_scale_factor
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=0.1)

        output = v3.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 25, 20)
key = torch.randn(1, 50, 20)
value = torch.randn(1, 50, 1)
