
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        qk = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(K.shape[-1])
        qk = qk + mask
        softmax = torch.nn.Softmax(dim=-1)
        attn_weight = softmax(qk)
        output = torch.matmul(attn_weight, V)
        return Q
# Inputs to the model
Q = torch.randn(1, 128, 56, 56)
K = torch.randn(1, 128, 56, 56)
V = torch.randn(1, 128, 56, 56)
d_model = 128
N_head = 4
mask = (torch.rand(1, 56)*10 > 3.5).unsqueeze(0).cuda().float()
