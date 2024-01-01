
class Model(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
 
    def forward(self, query, key):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = 1/math.sqrt(key.size(-1))
        softmax_qk = torch.nn.functional.softmax(qk * scale_factor, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model(p)

# Inputs to the model
query = torch.randn(1, 512, 90, 100)
key = torch.randn(1, 512, 120, 100)
value = torch.randn(1, 512, 120, 100)
