
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, qo, qi, qv, ko, ki, kv, mask, scale):
        qk = torch.matmul(qi.div(scale), ko.transpose(-2, -1))
        scaled_qk = qk.div(scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.25)
        output = torch.matmul(dropout_qk, ki)
        return output * mask * scale # Multiply the output of the attention mechanism by the scale factor, while applying the input mask.

# Initializing the model
m = Model()

# Inputs to the model
qo = torch.randn(1, 12, 512)
qi = torch.randn(1, 12, 512)
qv = torch.randn(1, 12, 512)
ko = torch.randn(1, 12, 512)
ki = torch.randn(1, 12, 512)
kv = torch.randn(1, 12, 512)
mask = torch.zeros((1, 1, 1))
scale = 1024.0
