
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        v6 = qk.mul(scale_factor)
        v7 = v6.softmax(dim=-1)
        v8 = torch.nn.functional.dropout(v7, p=dropout_p)
        output = v8.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 6, 128, 196)
key = torch.randn(1, 6, 196, 128)
value = torch.randn(1, 6, 196, 128)
scale_factor = 1
dropout_p = 0.5
