
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        