
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size):
        super().__init__()

 
    def forward(self, query, key, value, scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional().dropout(softmax_qk, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model(10, 20, 30)

# Inputs to the model
query = torch.randn(10, 20)
key = torch.randn(10, 30)
value = torch.randn(10, 30)
scale_factor = 0.015334055980135918
dropout_p = 0.37149580960278996
