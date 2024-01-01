
class Model(torch.nn.Module):
    def __init__(self, dim_query, dim_kv):
        super().__init__()
        self.scale_factor = 1 / (dim_kv**0.5)
 
    def forward(self, queries, keys, values, dropout_p):
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(values)
        return output

# Initializing the model
m = Model(dim_query=256, dim_kv=512)

# Inputs to the model
queries = torch.randn(1, 32, 256)
keys = torch.randn(1, 32, 512)
values = torch.randn(1, 32, 512)
dropout_p = 0.1
