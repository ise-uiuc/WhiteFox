
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        v = qk.div(0.125)
        softmax_v = v.softmax(dim=-1)
        dropout_v = torch.nn.functional.dropout(softmax_v, p=0.1)
        output = dropout_v.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 6, 128)
x2 = torch.randn(5, 128, 64)
