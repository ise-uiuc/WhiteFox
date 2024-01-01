
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)

        return output

# Initializing the model
m = Model()
scale_factor = 10.0
dropout_p = 0.2

# Inputs to the model
query = torch.randn(8, 6, 3, 5)
key = torch.randn(8, 1, 3, 7)
value = torch.randn(8, 1, 3, 5)
