
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, k, q):
        qc = q.size(-2)
        kc = k.size(-2)
        qr = q.size(-1)
        kr = k.size(-1)
        