
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 12, 512)
k = torch.randn(1, 512, 12)
v = k
inv_scale_factor = 10.0
dropout_p = 0.2
