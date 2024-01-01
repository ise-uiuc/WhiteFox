
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_1 = torch.nn.Linear(5, 10)
        self.model_2 = torch.nn.ReLU()
    
    def forward(self, x):
        y = self.model_1(x)
        y = self.model_2(y)
        return y
# Initializing the model
m = Model()

# Inputs to the Model
x1 = torch.randn(1, 5)
