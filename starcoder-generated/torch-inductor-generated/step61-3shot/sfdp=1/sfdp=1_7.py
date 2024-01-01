
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p, dropout_i):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 / inv_scale_factor
        s_qk = torch.nn.functional.softmax(v2, dim=-1)
        v3 = torch.nn.functional.dropout(s_qk, p=p, iid=d)
        o = i.matmul(v3)
        return o

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 512, 50)
key = torch.randn(3, 512, 18)
value = torch.randn(3, 512, 18)
inv_scale_factor = 1.0 / 0.5
dropout_p = 0.1
dropout_i = True
