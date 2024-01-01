
class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_p):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = torch.nn.ModuleList([torch.nn.Dropout(p=dropout_p) for _ in range(num_layers)])
 
        self.all = nn.Linear()

    def forward(self, x1):
        for layer in self.all:
          # do stuff here
        return v6

# Initializing the model
m = Model(...)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
