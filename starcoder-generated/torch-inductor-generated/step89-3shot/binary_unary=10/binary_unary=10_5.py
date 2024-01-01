
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(18432, 200)
        self.linear2 = torch.nn.Linear(200, 4)
 
    def forward(self, x1):
        print("Forward")
        v1 = self.linear1(x1)
        v2 = v1 + x1
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
input_x = torch.randn(2, 18432)
output = m(input_x)

if torch.cuda.is_available:
    x1 = torch.randn(1, 3, 64, 64).cuda()
    