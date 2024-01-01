
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.5
        v3 = self.linear(v2)  
        v4 = v1 - 0.45
        v5 = self.linear(v4)
        v6 = v1 - 0.405
        t1 = torch.min(torch.cat((v3.squeeze(0).unsqueeze(1), v5.squeeze(0).unsqueeze(1),v6.squeeze(0).unsqueeze(1)),1),1)[0]
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
