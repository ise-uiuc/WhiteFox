
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output, (scaled_qk, softmax_qk, dropout_qk)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 64)
k = torch.randn(8, 8, 64)
v = torch.randn(8, 8, 64)
inv_scale_factor=1.0
dropout_p=0.5
__output__, __state__ = m(q, k, v, inv_scale_factor, dropout_p)

