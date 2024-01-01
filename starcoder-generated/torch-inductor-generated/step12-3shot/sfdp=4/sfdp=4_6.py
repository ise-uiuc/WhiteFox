
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        q = torch.matmul(x1, x2)
        k = torch.matmul(x3, x4)
        v = k
        