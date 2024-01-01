
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
 
    def forward(self, queries, keys, values, scale_factor, dropout_p):
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(values)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(3, 4, 5)
keys = torch.randn(5, 4, 6)
values = torch.randn(5, 4, 6)
scale_factor = 5
dropout_p = 0.2
