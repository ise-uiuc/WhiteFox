
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(256, 256)
        self.linear2 = torch.nn.Linear(16, 16)
        self.linear3 = torch.nn.Linear(4, 4)
 
    def forward(self, x1):
        