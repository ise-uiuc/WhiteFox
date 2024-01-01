
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(128, 128)
        self.key = torch.nn.Linear(128, 128)
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        output = torch.matmul(q, k.transpose(-2, -1))
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
