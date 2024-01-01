
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 32)
 
    def forward(self, x1, **kwargs):
        v1 = self.fc(x1)
        # If other is in kwargs
        if 'other' in kwargs:
            v2 = kwargs['other'] + v1
        else:
            v2 = v1
        # If beta is in kwargs
        if 'beta' in kwargs:
            v3 = torch.sigmoid(v2 * kwargs['beta'])
        else:
            v3 = torch.sigmoid(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
