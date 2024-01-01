
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = [torch.randn(8, 8), torch.randn(8, 8)]
    def forward(self, Q, K, V, mask):
        weightQ, weightK = self.weights
        qk = X @ K.t() / math.sqrt(X.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 8, 56, 56)
K = torch.randn(1, 8, 56, 56)
V = torch.randn(1, 8, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
