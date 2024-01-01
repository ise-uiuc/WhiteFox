
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries, keys, values)
        queries = torch.nn.functional.normalize(queries, p=2, dim=-1)
        keys = torch.nn.functional.normalize(keys, p=2, dim=-1)
        qk = torch.matmul(queries, keys.transpose(-2 -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(values)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.rand(16, 12, 8)
keys = torch.rand(16, 28, 8)
values = torch.rand(16, 28, 32)
