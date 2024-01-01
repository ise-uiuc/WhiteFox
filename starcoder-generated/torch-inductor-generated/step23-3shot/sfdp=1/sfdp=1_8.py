
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dropout_p = 0.2
m = Model()

# Inputs to the model
__q__ = torch.randn(3, 4, 32)
__k__ = torch.randn(3, 4, 64)
__v__ = torch.randn(3, 4, 64)
scale = float(2 ** 0.5)
