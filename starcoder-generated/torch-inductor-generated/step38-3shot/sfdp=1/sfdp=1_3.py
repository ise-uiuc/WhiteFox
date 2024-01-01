
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(0.0625)
        v3 = v2.softmax(dim=-1)
        v4 = F.dropout(v3, p=0.3)
        v5 = torch.matmul(v4, x3)
        