
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 5, 8)
key = torch.randn(2, 8, 5) # `k*q` is the dimension of query and key, `key.transpose(-2, -1)` is the dimension of key and value, so `(key.transpose(-2, -1)).shape = (2, 5, 5)`
value = torch.randn(2, 8, 5)
inv_scale_factor = 1 / np.sqrt(np.prod(key.shape[-2:]))
dropout_p = 0.2
