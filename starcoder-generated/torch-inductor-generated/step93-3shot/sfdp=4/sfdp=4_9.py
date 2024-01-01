
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Linear(in_features=64, out_features=64)
        self.w1 = torch.nn.Linear(in_features=64, out_features=64)
        self.w2 = torch.nn.Linear(in_features=128, out_features=128)
        self.w3 = torch.nn.Linear(in_features=128, out_features=128)
    def forward(self, qk, v0, mask):
        qk0 = self.w0(qk)
        qk1 = torch.tanh(qk0)
        qk2 = self.w1(qk1)
        qk3 = torch.tanh(qk2)
        v00 = self.w2(v0)
        v01 = torch.tanh(v00)
        v02 = self.w3(v01)
        v03 = torch.tanh(v02)
        qkv00 = torch.cat([qk3, v03], -1)
        qkv01 = self.w4(qkv00)
        qkv02 = torch.tanh(qkv01)
        attn_weight = torch.softmax(qkv02, dim=-1)
        return attn_weight
# Inputs to the model
Q0 = torch.randn(1, 64, 56, 56)
K1 = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 128, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
