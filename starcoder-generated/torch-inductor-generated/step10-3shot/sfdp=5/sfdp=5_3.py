
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5, bias=False)
 
    def forward(self, t1, t2, t3):
        t4 = self.linear(t2)
        t5 = torch.einsum('bqa,bbqc->bqc', t1, t4)
        t6 = t3 + t5
        return t6

# Initializing the model
m = Model()

# Inputs to the model
t1 = torch.randn(1, 5, 10, 20)
t2 = torch.randn(8, 5)
t3 = torch.randn(1, 5, 10, 20)
