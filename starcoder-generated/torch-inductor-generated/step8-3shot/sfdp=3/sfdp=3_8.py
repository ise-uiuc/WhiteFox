
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

# Inputs to the model
scale_factor = torch.rand(2, 2)
dropout_p = torch.rand(1)
x1 = torch.randn(2, 2, 4, 8)
x2 = torch.randn(2, 3, 4, 4)
q = torch.randn(2, 3, 64, 64)
