
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(15, 1)
        self.linear2 = torch.nn.Linear(15, 5)

    def forward(self, k, v, n, q):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk.masked_fill(n == 1, float('-inf'))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output

# Initializing the model
m = Model()
k = torch.randn(3, 4, 5)
v = torch.randn(3, 4, 10)
n = torch.ones(3, 4)
n[[0, 1], :] = 1
n[:, [0, 2, 3]] = 1
qk = torch.randn(3, 3, 10)
