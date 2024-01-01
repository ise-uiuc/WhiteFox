
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1, x2):  # x2 is the "other" tensor
        v1 = self.linear(x1)
        