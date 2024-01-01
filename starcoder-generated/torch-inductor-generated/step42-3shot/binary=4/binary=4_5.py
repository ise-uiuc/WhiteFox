
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        linear = torch.nn.Linear(8, 8)
        linear.weight.data.fill_(1.5)
        self.linear = linear
 
    def forward(self, x1):
        