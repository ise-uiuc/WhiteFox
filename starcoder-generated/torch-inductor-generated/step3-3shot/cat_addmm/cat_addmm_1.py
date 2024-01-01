
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(64, 128)
        self.lin2 = torch.nn.Linear(128, 3)
 
    def forward(self, input):
        mat1 = self.lin1(input)
        mat2 = self.lin2(mat1)
        return torch.cat([input, mat2], dim=-1)

# Initializing the model
m = Model()
 
# Inputs to the model
input_tensor = torch.randn(32, 64)
