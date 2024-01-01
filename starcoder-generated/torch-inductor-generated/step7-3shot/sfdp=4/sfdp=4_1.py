
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=12, out_features=24, bias=True)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=24, out_features=56, bias=True)
 
    def forward(self, data):
        data = self.linear1(data)
        data = self.relu(data)
        data = self.linear2(data)
        return data

# Initializing the model
m = Model()

# Inputs to the model
data = torch.randn(10, 12)
