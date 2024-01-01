
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.nn.Linear(128, 128, bias=False)
        self.matmul2 = torch.nn.Linear(128, 128, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.matmul1(x1).softmax(-1).matmul(x2)# torch.matmul(x1.softmax(-1), x2)
        v2 = self.matmul2(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 128)
x2 = torch.randn(64, 128)
