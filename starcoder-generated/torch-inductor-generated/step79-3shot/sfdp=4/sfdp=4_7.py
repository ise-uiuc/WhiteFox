
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#     def forward(self, Q, K, V, M):
#         qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
#         qk = qk + M
#         attn_weight = torch.softmax(qk, dim=-1)
#         output = attn_weight @ V
#         return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
M = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
