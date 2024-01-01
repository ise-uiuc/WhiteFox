
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define and initialize linear layers
        self.linear1 = torch.nn.Linear(100, 100, bias=True)
        self.linear2 = torch.nn.Linear(100, 100, bias=True)
        self.linear3 = torch.nn.Linear(100, 1, bias=True)

        for layer in self.layers.values():
            if isinstance(layer, torch.nn.modules.Linear):
                torch.nn.init.uniform_(layer.weight)
                torch.nn.init.uniform_(layer.bias)
    def forward(self,x1):
        # Perform the forward pass
        v1 = self.activation_fc1(self.linear1(x1))
        v2 = self.activation_fc2(self.linear2(v1))
        return self.linear3(v2)
# Inputs to the model
x1 = torch.randn(2, 100)
