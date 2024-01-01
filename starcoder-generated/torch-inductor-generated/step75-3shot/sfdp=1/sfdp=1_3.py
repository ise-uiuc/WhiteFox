
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        q = torch.nn.functional.linear(x1, x2)
        k = torch.nn.functional.linear(x3, x2)
        k = k.transpose(-2, -1)
        v = torch.nn.functional.linear(x3, x2)
        v = v.transpose(-2, -1)
        qk = q.matmul(k)
        scale_factor = qk.size(-1) ** -0.25
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.125, training=True)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 28)
x2 = nn.Parameter(torch.randn(4, 10))
x3 = torch.randn(1, 10, 28)
