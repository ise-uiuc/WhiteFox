
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        qk = torch.matmul(x1, x1.transpose(-2, -1))
        inv_scale_factor = torch.sqrt(torch.tensor(x1.size(-1)))
        softmax_qk = qk.div(inv_scale_factor).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, 0.05)
        output = dropout_qk.matmul(x1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
