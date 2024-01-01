
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(
            nn.Linear(10, 24), # nn.Linear is a linear transformation
            lambda x: x + torch.tensor([0.98]), # x + torch.tensor([0.98]) adds torch.tensor([0.98]) to the output of the linear transformation
        )

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
