
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(8, 16)
        self.query = torch.nn.Linear(8, 16)
        self.value = torch.nn.Linear(8, 16)
 
    def forward(self, x2):
        k = self.key(x2)
        q = self.query(x2)
        v = self.value(x2)
        qk = q @ k.transpose(-2, -1)
        return qk

# Inputs to the model
x2 = torch.randn(1, 8, 10)
