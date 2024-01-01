
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = qk.shape[-1]
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return q, k, v, dropout_p, output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 2, 3)
k = torch.randn(2, 3, 4)
v = torch.randn(2, 3, 4)
dropout_p = 0.5
