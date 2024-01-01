
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, scale_factor, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 384, 24)
key = torch.randn(4, 24, 384)
scale_factor = torch.tensor(1.0 / math.sqrt(24))
value = torch.randn(4, 24, 384)
dropout_p = 0.0
