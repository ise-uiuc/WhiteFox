
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 15)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, self.linear.weight)
        v2 = v1.softmax(-1)
        v3 = torch.nn.functional.dropout(v2, p=0.2)
        v4 = v3.matmul(x2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 21, 15)
