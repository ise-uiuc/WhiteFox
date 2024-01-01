
def model1():
    model = torch.nn.Sequential(torch.nn.Linear(2, 4),
                                torch.nn.ReLU(),
                                torch.nn.Linear(4, 1))
    return model

# Initializing the input tensor
x1 = torch.randn(1,2)

# Inputs to the model
