
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, queries, keys, values, scale_factor, dropout_p):
        query = queries
        key = keys
        value = values
        inv_scale_factor = 1.0 / scale_factor
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(10, 20, 96)
keys = torch.randn(8, 20, 128)
values = torch.randn(8, 20, 128)
scale_factor = torch.randn((1, 1)).item()
dropout_p = torch.randn((1, )).item()
