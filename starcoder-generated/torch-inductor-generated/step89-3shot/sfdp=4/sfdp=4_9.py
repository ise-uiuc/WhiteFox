
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qn, kn, vn, bias):
        qk = qn @ kn.transpose(-2, -1)
        qk = qk + bias
        attn_weight = torch.softmax(qk, -1)
        output = attn_weight @ vn
        return output
# Inputs to the model
n1 = torch.randn(1, 56, 56, 64)
n2 = torch.randn(1, 56, 56, 64)
n3 = torch.randn(1, 56, 56, 64)
bias = torch.randn(1, 56, 56)
