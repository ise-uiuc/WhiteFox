
  class Model(torch.nn.Module):
      __init__(self):
          super().__init__()
          self.linear = torch.nn.Linear(9, 1)
     
      def forward(self, x1):
          v1 = self.linear(x1)
          return v1 - other
__call__():
      x = input_tensor + other
      return self.linear(x)

# Initializing the model
  m = Model()
# Inputs to the model
  x1 = torch.randn(1, 9)
  other = torch.randn(1, 1)
