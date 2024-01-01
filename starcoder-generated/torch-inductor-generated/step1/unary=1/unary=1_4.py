
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(224 * 224 * 3, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 1000)
        self.fc4 = torch.nn.Linear(1000, 1000)
        self.fc5 = torch.nn.Linear(1000, 1000)
        self.fc6 = torch.nn.Linear(1000, 1000)
        self.fc7 = torch.nn.Linear(1000, 1000)
        self.fc8 = torch.nn.Linear(1000, 1)
 
    def forward(self, x):    
        v1 = self.fc1(x)
        v2 = v1 * 0.5
        v3 = self.fc2(v1)
        v4 = v1 * 0.7978845608028654
        v5 = v4 + 1
        v6 = torch.tanh(v5)
        v7 = v3 * v6
        v8 = self.fc3(v7)
        v9 = v7 * 0.044715
        v10 = v7 * v9
        v11 = self.fc4(v10)
        v12 = v10 * v11
        v13 = self.fc5(v12)
        v14 = v12 * v13
        v15 = self.fc6(v14)
        v16 = v14 * v15
        v17 = self.fc7(v16)
        v18 = v16 * v17
        v19 = self.fc8(v18)
        return v19