
class BiasAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(64, 512), torch.nn.Linear(512, 256), torch.nn.Linear(256, 64), torch.nn.Linear(64, 64))
 
    def forward(self, input_data):
        x = input_data
        x = self.layers(x)
        return x

# Initializing the model
m = BiasAddModel()

# Inputs to the model
input_data = torch.randn(1, 64)
