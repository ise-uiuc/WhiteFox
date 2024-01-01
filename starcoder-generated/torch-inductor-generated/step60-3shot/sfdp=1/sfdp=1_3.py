
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 1.0 / math.sqrt(float(x1.shape[-1]))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(x1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand((5, 10))
x2 = torch.rand((3, 15))
