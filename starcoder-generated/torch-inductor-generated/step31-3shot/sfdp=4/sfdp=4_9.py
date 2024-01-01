
class ModelWithCustomMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(64, 56)
    def my_custom_linear_op(self, x):
        y = self.layer1(x)
        return y + 10
    def forward(self, q, k, v4, mask):
        q = self.my_custom_linear_op(q)
        k = self.my_custom_linear_op(k)
        v = self.my_custom_linear_op(v4)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v4
        return output
# Inputs to the model
Q = torch.randn(1, 64)
K = torch.randn(1, 64)
V = torch.randn(1, 64)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
