
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.nn.Linear(in_features=720, out_features=720)
        self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self, x1):
        v1 = self.matmul(x1)
        v2 = self.softmax(v1)
        v3 = torch.nn.functional.dropout(v2, p=0.1)
        v4 = v3.mul(0.125)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 720)
