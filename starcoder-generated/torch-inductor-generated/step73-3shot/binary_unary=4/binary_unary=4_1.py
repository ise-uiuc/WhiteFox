 
class Model(torch.nn.Module):
    def forward(self, x1, other):
        v1 = torch.nn.functional.linear(x1, torch.ones(1, 25)) + other
        return torch.nn.functional.relu(v1)

# Initializing the model and a keyword argument
other = torch.ones(1, 25)
m = Model()

# Call the model. The keyword argument is passed to the forward function so that it can be applied to other.
