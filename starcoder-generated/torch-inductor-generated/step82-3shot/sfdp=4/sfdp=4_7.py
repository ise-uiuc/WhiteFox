
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.qk = torch.nn.Parameter(torch.randn(1, 1, 64, 64))
    def forward(self, Q7, K, V, mask):
        qk = self.qk @ K.transpose(-2, -1)
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q7 = torch.randn(1, 1, 64, 64)
K2 = torch.randn(1, 1, 64, 64)
V = torch.randn(1, 1, 64, 64)
mask2 = (torch.rand(1, 1, 64, 64) > 0.5).fill_(-1000000000.0)
