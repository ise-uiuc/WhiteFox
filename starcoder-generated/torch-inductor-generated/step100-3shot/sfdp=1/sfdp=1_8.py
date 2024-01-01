
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, qi, ki, v):
        qk = torch.matmul(qi, ki.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
qi = torch.randn(8, 6, 256)
ki = torch.randn(8, 6, 256)
v = torch.randn(8, 6, 64)
