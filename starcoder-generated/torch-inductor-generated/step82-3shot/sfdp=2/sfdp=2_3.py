
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, key_pad, query, value_pad):
        v1 = torch.matmul(query, key_pad.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = v4.matmul(value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
key_pad = torch.randn(1, 3, 8, 16)
query = torch.randn(1, 3, 8, 16)
value_pad = torch.randn(1, 3, 8, 128)
