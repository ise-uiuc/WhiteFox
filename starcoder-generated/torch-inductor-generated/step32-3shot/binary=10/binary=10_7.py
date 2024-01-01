
class ModelFactory:
    @staticmethod
    def build_model():
        return torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.Linear(64, 128))
 
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_factory = ModelFactory()

    def forward(self, x1):
        return self.model_factory.build_model()(x1)

# Initializing the model
m = MyModel()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
