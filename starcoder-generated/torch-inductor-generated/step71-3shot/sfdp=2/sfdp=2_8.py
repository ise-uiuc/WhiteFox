
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, x2):
        q = torch.randn(64, 8, 10, 10)
        k = torch.randn(64, 8, 10, 10)
        v = torch.randn(64, 8, 10, 10)
        qk = q.matmul(k.transpose(2, 3))
        