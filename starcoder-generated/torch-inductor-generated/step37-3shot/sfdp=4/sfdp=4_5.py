
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 32)
        torch.nn.init.normal_(self.fc1.weight, std=0.01)
        self.fc2 = torch.nn.Linear(32, 64)
    def forward(self, Q0, K0, V0, mask):
        q = self.fc1(Q0)
        k = self.fc2(K0)
        v = self.fc1(V0)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output, qk
# Inputs to the model
Q = torch.randn(2, 2)
K = torch.randn(2, 2)
V = torch.randn(2, 2)
mask = (torch.rand(2, 2) > 0.7).fill_(-1000000000.0)
