
class Model():
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(__in_features__, __out_features__, bias=True)
 
      def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - __other__
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(__in_features__)
