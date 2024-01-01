
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, x):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(x)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, x)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q, k, v = torch.randn(1, 512, 4, 64), torch.randn(1, 512, 64, 32), torch.randn(1, 512, 32, 4)
x = torch.randn(1)
