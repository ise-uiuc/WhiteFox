
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = softmax_qk.dropout(p=dropout_p).matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 100, 100)
key = torch.randn(1, 100, 256)
value = torch.randn(1, 256, 1)
inv_scale_factor = 2.0
dropout_p = 0.0
