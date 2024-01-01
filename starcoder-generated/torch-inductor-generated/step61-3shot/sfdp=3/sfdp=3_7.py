
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, queries, keys, value, scale_factor, dropout_p):
        v1 = torch.matmul(queries, keys.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(16, 8, 512)
keys = torch.randn(16, 16, 512)
value = torch.randn(16, 16, 1024)
scale_factor = torch.FloatTensor([512].float())
dropout_p = 0.1
