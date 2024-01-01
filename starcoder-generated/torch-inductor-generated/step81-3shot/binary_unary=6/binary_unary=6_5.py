
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 16
        out_features = 8
        hidden_layer_size = 100
        self.linear1 = torch.nn.Linear(in_features, hidden_layer_size, bias=False)
        self.linear2 = torch.nn.Linear(hidden_layer_size, out_features, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.ReLU()(x)
        x = self.linear2(x)
        x = torch.add(x, self.other)
        return x

# Initializing the model
m = Model()

#Inputs to the model
self.other = torch.nn.Parameter(torch.randn(8))
x = torch.randn(2, 16)
