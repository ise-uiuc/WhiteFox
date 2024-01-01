
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, value):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = qk.size(-1) ** -0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.0)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 4, 64, 64)
k = torch.randn(1, 4, 64, 64)
value = torch.randn(1, 4, 64, 64)
