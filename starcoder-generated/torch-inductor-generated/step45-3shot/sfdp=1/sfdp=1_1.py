
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, in_scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(in_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 16, 4)
k = torch.randn(1, 32, 4)
v = torch.randn(1, 32, 4)
in_scale_factor = 10.
