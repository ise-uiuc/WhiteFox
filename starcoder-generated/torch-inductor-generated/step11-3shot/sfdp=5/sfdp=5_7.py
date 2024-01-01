
L = 8
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(...)
        self.fc2 = torch.nn.Linear(...)
        self.fc3 = torch.nn.Linear(...)
        self.fc4 = torch.nn.Linear(...)
 
    def forward(self, x1, x2, x3, mask):
        v1 = self.fc1(x1)
        v2 = self.fc2(x2)
        v3 = self.fc3(x3)
        v4 = self.fc4(torch.cat((v1, v2, v3), -1))
        v5 = torch.softmax(v4, dim=-1)
        output = v5.masked_fill((mask == 0), float('-inf'))
        return output

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(L, D1)
x2 = torch.randn(L, D2)
x3 = torch.randn(L, D3)
mask = torch.randint(2, L).unsqueeze(-1)
