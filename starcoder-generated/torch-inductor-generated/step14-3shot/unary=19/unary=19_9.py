
class Model(torch.nn.Module):
    def __init__(self, input_features, output_features):
       super(Model, self).__init__()
       self.mlp = torch.nn.Sequential(
           torch.nn.Linear(input_features, 16),
           torch.nn.Linear(16, 8),
           torch.nn.Linear(8, output_features),
       )
 
    def forward(self,x1):
        return self.mlp(x1)

# Initializing the model
m = Model(100, 4)
# Inputs to the model
x1 = torch.randn(1000, 100)
