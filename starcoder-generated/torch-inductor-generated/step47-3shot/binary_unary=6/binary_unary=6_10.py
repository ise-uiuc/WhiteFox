
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6*2*2, 4)
        self.other = torch.nn.Parameter(torch.tensor([0.5, 0.6, 0.7, 0.8], dtype=torch.float))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Initializing the 'other' variables of the model
# 'other' is used in the model
torch.nn.init.constant_(m.other, 1)

# Inputs to the model
x1 = torch.randn([1,6,2,2])
