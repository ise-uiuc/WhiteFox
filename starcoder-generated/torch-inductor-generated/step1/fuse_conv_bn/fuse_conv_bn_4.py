
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(2, 2, 3)
        self.batch_norm = torch.nn.BatchNorm1d(2)
    
    def forward(self, x):
        # In the first scenario
        x1 = self.conv1d(x)
        y1 = self.batch_norm(x1)

        # In the second scenario
        x2 = self.conv1d(x)
        y2 = torch.nn.functional.batch_norm(x2, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight, self.batch_norm.bias, False, 0.1)
        
        # Return just the linear value in each scenario
        return x1, x2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 8)
