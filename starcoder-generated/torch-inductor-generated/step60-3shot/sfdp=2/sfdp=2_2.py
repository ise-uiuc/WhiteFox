
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, queries, keys, values, mask=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.ones((1, 3, 16))
keys = torch.ones((1, 6, 16))
values = torch.ones((1, 6, 16))
mask = torch.ones((1, 3, 6))
