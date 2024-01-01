
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 16)
 
        def forward(self, x1):
            y = self.fc(x1)
            y = y * 0.5
            y = y + (y*y*y) * 0.044715
            y = y * 0.7978845608028654
            y = torch.tanh(y)
            y = y + 1
            y = y * y
            return y
 
# Initializing the model
m = Model()

# Initializing the model
