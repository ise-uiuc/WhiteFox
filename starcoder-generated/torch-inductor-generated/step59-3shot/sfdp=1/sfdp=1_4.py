
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, queries, keys, values):
        v1 = torch.matmul(queries, keys.transpose(-2, -1))
        v2 = v1.div(0.01 * math.sqrt(queries.shape[-1]))
        v3 = v2.softmax(dim = -1)
        dropout_v3 = torch.nn.functional.dropout(v3, p=0.1)
        v4 = torch.matmul(dropout_v3, values)
        return v4
 
# Initializing the model
m = Model()
 
# Inputs to the model
queries = torch.randn(1, 5, 8)
keys = torch.randn(1, 10, 8)
values = torch.randn(1, 10, 8)
