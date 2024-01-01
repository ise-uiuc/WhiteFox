
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        q = x1
        k = x2
        v = x2
        w = torch.zeros(q.size())
        