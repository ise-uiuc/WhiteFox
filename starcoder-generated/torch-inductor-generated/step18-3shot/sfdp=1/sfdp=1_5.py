
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
q = torch.randn(1, 128, 56, 56)
k = torch.randn(1, 128, 32, 32)
v = torch.randn(1, 128, 32, 32)
inv_scale_factor = 0.5
dropout_p = 0.2
