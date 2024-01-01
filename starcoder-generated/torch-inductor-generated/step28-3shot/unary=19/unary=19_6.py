
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_module = torch.nn.Sequential(
          torch.nn.Linear(25*25,1024),
          torch.nn.LeakyReLU(0.2,inplace=True), # Apply the Leaky ReLU function to the output of linear transformation
          torch.nn.Linear(1024,1)
        )
 
    def forward(self, x1):
        v1 = self.mlp_module(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25*25)
