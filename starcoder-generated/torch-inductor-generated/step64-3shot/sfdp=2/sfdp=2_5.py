
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2)
        inv_scale_factor = math.sqrt(x1.size(-1))
        softmax_qk = qk.div(inv_scale_factor).softmax(dim=-1)
        dropout_p = 0.1
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
