
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(query, key, value, scale, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
scale = 5
dropout_p = 0.2
