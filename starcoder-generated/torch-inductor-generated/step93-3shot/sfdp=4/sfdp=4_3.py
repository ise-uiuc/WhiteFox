
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q0, k, v):
        qk = q0 @ k.transpose(-2, -1) / math.sqrt(q0.size(-1))
        output = torch.softmax(qk, dim=-1)
        return output
# Inputs to the model
qk = torch.randn(1, 64, 56, 56)
