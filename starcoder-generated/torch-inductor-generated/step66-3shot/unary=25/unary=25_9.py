
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(64, 64, bias=True)
        self.linear_2 = torch.nn.Linear(64, 64, bias=False)
 
    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = x1 >= 0
        x3 = x1 * 0.1
        x4 = torch.where(x2, x1, x3)
        x5 = self.linear_2(x4)
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)


for i in range(30):
    result = m(x * i)

# Print results
print(result)

