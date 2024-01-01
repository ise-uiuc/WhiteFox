
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, c, mask):
        qk = a @ b.transpose(-2, -1) # 128, 2048, 2048
        t1 = qk / math.sqrt(qk.size(-1)) + mask # 128, 2048, 2048 * 128, 2048, 2048 + 128, 2048, 2048
        t2 = torch.softmax(t1, dim=-1) # 128, 2048, 2048
        output = t2 @ c # 128, 2048, 2048 * 128, 2048, 2048
        return output
# Inputs to the model
Q = torch.randn(1, 128, 2048)
K = torch.randn(1, 128, 2048)
V = torch.randn(1, 128, 2048)
mask = torch.randn(1, 128, 2048)
mask[mask>=0.7] = float("-inf")
mask[mask<=-0.7] = float("inf")
mask = torch.softmax(mask, dim=-1)
