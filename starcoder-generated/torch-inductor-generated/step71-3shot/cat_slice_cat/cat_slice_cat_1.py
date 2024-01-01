
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        x = [x1, x2]
        x1, x2 = torch.split(torch.cat(x, dim=1), [5, 7], dim=1)
        x = [x1, x2]
        return torch.cat(x, dim=1)
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 5)
x2 = torch.randn(2, 7)
