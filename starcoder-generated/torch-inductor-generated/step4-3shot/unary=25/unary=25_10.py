
class Model(torch.nn.Module):
     def __init__(self, negative_slope=0.01):
          super(Model, self).__init__()
          self.linear = torch.nn.Linear(64, 128)
          self.negative_slope = torch.nn.parameter.Parameter(
               torch.tensor(
                   negative_slope
               )
           )
 
     def forward(self, x1,):
         v1 = torch.nn.functional.leaky_relu(
             self.linear(x1),
             negative_slope=self.negative_slope
         )
         return v1
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 64)
