
class Model(torch.nn.Module):
    def __init__(self, max_value, min_value):
        super().__init__()
        self.linear = torch.nn.Linear(1, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        