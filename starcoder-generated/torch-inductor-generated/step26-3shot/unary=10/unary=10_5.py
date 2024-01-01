
class Model(torch.nn.Module):
    def __init__(self):
        self.linear1 = LinearLayer(1024, 1024)
 
    def forward(self, l):
        l1 = self.linear1(l)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        return l5
 
# Initializing the model
m = Model()

# Inputs to the model
l = torch.ones(1, 1024)
