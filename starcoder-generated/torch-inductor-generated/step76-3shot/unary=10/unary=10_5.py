
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = torch.nn.Linear(16,32,bias=True)
 
    def forward(self, x1):
        x4=self.f1(x1)
        x6 = x4 + 3
        x8 = torch.clamp_min(x6, 0)
        