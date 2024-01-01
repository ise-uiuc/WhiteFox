
class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(12, 13), torch.nn.ReLU(), torch.nn.Linear(13, 13), torch.nn.ReLU(), torch.nn.Linear(13, 13), torch.nn.ReLU(), torch.nn.Linear(13, 17))


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 12)
