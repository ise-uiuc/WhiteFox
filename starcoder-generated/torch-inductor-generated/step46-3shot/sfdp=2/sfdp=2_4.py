
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key,value,inv_scale_factor,dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        output = v4.matmul(value)

        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 8)
key = torch.randn(1, 12, 8)
value = torch.randn(1, 12, 10)
inv_scale_factor = torch.tensor([0.5])
dropout_p = torch.tensor([0.5])
