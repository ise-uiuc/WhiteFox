
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Set hidden size to 1 and 2
        self.layers = nn.Linear(1, 2)
        
        if (False):
            # Add a dropout layer
            self.dropout = nn.Dropout(p=1.0)
    
    def forward(self, x):
        x = self.layers(x)

        if True:
            # Perform a concatination operation
            x = x.flatten(start_dim=1)

        return x
# Inputs to the model
x = torch.randn(2, 1)
