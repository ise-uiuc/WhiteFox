
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
   
    def forward(self, x1):
        v1 = torch.mm(x1, torch.rand((100, 640), dtype=torch.float32))
        