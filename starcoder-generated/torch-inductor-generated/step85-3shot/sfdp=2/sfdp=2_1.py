
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q0, k1, v0, inv_scale):
        qk = torch.matmul(q0, k1.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=1.0)
        output = dropout_qk.matmul(v0)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
q0 = torch.randn(15, 25, 20)
k1 = torch.randn(15, 25, 30)
v0 = torch.randn(15, 25, 30)
inv_scale= 1
