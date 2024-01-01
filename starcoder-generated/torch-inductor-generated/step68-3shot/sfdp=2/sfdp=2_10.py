
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(512, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
 
  def forward(self, _input):
        v1 = F.relu(self.fc1(input))
        v3 = v1.abs()
        v4 = self.fc2(v3)
 
        return v4

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 512)
