
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = x1 @ x2.transpose(-2, -1)
        v2 = v1 / math.sqrt(v1.size(-1))
        v3 = v2 + x3.float().masked_fill(x3 == float("-inf"), -1e9)
        v4 = torch.softmax(v3, dim=-1)
        