
class Model(torch.nn.Module):
    def forward(self, x, y):
        v1 = torch.cat((x, y), 1)
        v2 = torch.cat((v1, v1[0: None]), dim=0)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
