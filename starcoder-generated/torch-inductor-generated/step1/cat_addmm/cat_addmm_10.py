
class Model(torch.nn.Module):
    def forward(self, x):
        s1 = torch.addmm(x, x, x, beta=0.0, alpha=0.0)
        s2 = torch.addmm(x, x, x, beta=0.0, alpha=0.0)
        return torch.cat((s1, s2), -1) # or return s1 + s2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(10, 10)
