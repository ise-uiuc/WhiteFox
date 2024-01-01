
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v3, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight_1 = torch.softmax(qk, dim=-1)
        output_1 = attn_weight_1 @ v3.unsqueeze(-2)*attn_weight_1.unsqueeze(1).unsqueeze(-1)
        output_1 = torch.sum(output_1, dim=2)
        output = output_1.squeeze(-1)
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
