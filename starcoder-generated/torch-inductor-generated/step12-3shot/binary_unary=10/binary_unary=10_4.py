
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=16, out_features=hidden_size)
        self.linear2 = torch.nn.Linear(in_features=hidden_size, out_features=1)
 
    def forward(self, x):
        l1 = self.linear1(x)
        l2 = self.linear2(l1)
        l3 = l1.sigmoid()
        l4 = l2.sigmoid()
        l5 = l3 * l4
        return l5

# Initializing the model with a hidden size of 128
m = Model(128)

# Inputs to the model
x = torch.randn(8, 16)
