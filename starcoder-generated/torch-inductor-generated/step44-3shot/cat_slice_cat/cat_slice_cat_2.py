
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), dim=1)
        size = x2.size()[1]
        v2 = v1[:, :size]
        v3 = torch.cat([v1, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.ones(1, 2, 2)
x2 = torch.ones(1, 4, 2)
