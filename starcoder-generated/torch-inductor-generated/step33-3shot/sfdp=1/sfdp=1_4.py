
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1.0 / math.sqrt(query.shape[-1])
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(value)
        return output

# Initializing the module
m = Model()

# Inputs to the module
query = torch.randn(4, 196, 768)
key = torch.randn(4, 256, 768)
value = torch.randn(4, 256, 768)
