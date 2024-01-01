
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 512, 16, 16)
k = torch.randn(1, 512, 16, 16)
v = torch.randn(1, 512, 16, 16)
scale_factor = 0.5
dropout_p = 0.0
