
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(10, 20)
        self.linear_2 = torch.nn.Linear(20, 30)

    def forward(self, hidden):
        t = F.relu(self.linear_1(hidden))
        t1 = self.linear_2(t)
        t2 = t1[:,1,:,:]
        t3 = t2 * 0.5
        t4 = t2 * 0.7071067811865476
        t5 = torch.erf(t4) + 1
        t6 = - t3 * t5
        return self.linear_2(t6)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 10)
