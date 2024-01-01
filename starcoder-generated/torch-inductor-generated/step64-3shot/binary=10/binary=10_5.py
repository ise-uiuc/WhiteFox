
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, other)
        return v1

# Initializing the model
m = Model()

