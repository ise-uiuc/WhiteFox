
class Model(torch.nn.Module):
    def forward(self, in0, in1):
        in2 = torch.cat([in0, in1], dim=1)
        v0 = in2.view(3, 2, 2)
        v1 = F.relu(v0)
        v2 = torch.cat([v1, -v1], dim=1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 2)
