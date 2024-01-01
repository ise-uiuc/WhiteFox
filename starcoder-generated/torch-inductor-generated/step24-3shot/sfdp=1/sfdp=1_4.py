
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.sum(qk, dim=-1, keepdim=True))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 5, 100)
key = torch.randn(2, 4, 100)
value = torch.randn(2, 4, 16384)
dropout = 0.1
