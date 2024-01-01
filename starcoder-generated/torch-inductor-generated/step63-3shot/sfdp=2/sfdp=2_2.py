
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.rand(128, 128)
    
    def forward(self, x1, x2):
        v = torch.matmul(x1, x2.t())
        v = v.div(self.weight)
        v = v.softmax(dim=-1)
        v = torch.nn.functional.dropout(v, p=0.2)
        # 