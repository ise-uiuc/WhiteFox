
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4, bias = False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        a1 = torch.tensor([0.486427, 0.643821, 0.344947, 0.593710])
        