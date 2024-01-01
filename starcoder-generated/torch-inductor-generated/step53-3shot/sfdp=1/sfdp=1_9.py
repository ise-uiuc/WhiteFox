
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, queries, keys, values, scale_factor, dropout_p):
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, values)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 2, 20)
keys = torch.randn(1, 2, 40)
values = torch.randn(1, 2, 40)
scale_factor = torch.randn(1, 1)
dropout_p = 0.5
