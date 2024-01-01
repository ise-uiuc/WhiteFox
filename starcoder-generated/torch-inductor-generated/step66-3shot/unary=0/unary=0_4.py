
class MyModule2(torch.nn.Module):
  def __init__(self, n_states, n_hidden, n_layers):
    super().__init__()
    self.all_layers = []
    self.n_states = n_states
    self.n_layers = n_layers
    
    self.l_i = torch.nn.Linear((self.n_states), n_hidden)

    for i in range(self.n_layers):
        self.all_layers.append(torch.nn.Linear(n_hidden, n_hidden))


  def forward(self, x):
    x = torch.tanh(self.l_i(x))
    for i in range(self.n_layers):
      x = torch.tanh(self.all_layers[i](x))
    #return x
    return torch.tanh((self.all_layers[-1](x))) 

class MyModule(torch.nn.Module):
    def __init__(self, n_states, n_hidden, n_layers):
        super().__init__()
        self.mlp= MyModule2(n_states, n_hidden, n_layers)
        self.conv= torch.nn.Conv2d(n_hidden, 1, 1, stride=1, padding=0)

    def forward(self, x):
        v1 = self.mlp(x)
        v2 = self.conv(v1)
        return v2

# Inputs to the model
x4 = torch.randn(1, 12, 34, 12)
