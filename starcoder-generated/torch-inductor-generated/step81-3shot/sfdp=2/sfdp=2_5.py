
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 768)
 
    def forward(self, x1, x2):
        q = self.linear(x1)
        k = self.linear(x2)
        s = torch.matmul(q, k.transpose(-2, -1))
        v = self.linear(x2)
        i = 2.2
        return s.div(i).softmax(dim=-1).matmul(v)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 128)
