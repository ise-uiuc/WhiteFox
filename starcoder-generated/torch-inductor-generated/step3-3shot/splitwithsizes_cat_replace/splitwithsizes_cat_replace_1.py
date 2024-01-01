
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1 + torch.split(input=x1, split_size_or_sections=[1, 2], dim=1)
        v2 = torch.cat([v1[0], v1[2]], dim=1)
        return torch.split(input=v2, split_size_or_sections=[1, 2], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(size=[1, 3, 64, 64])
