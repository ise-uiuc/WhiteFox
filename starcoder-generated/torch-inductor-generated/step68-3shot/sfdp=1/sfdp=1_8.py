
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(768, 768)
        self.key = torch.nn.Linear(768, 768)
        self.value = torch.nn.Linear(768, 768)
 
    def forward(self, x1, x2, x3):
        w1 = self.query(x1).softmax(dim=-1)
        w2 = self.key(x2)
        w3 = self.value(x3)
        v1 = w1.matmul(w2).matmul(w3)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768)
x2 = torch.randn(1, 768)
x3 = torch.randn(1, 768)
