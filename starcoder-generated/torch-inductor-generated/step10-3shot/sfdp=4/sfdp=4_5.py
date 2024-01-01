
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = x1 @ x2.transpose(-2, -1) / math.sqrt(32)
        v2 = v1 + -1e10*(x1 == x2)
        v3 = torch.softmax(v2, -1)
        