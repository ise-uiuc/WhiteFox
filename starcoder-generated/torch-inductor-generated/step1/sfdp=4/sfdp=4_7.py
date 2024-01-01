
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(8, 16)
 
    def forward(self, x):
        q = self.linear1(x)
        k = self.linear2(x)
        v = torch.empty(5, 4, 16)
        attn_mask = torch.empty(5, 4, 4, dtype=bool)
