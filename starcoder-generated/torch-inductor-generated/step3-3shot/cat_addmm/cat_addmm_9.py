
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.m00 = torch.nn.Linear(in_features=224, out_features=10)
        self.m10 = torch.nn.Linear(in_features=30, out_features=10)

    def input_func(self, input):
        return torch.cat([input[0], input[1]], dim=1)

    def forward(self, x):
        x0 = self.m00(x.narrow(1, 0, 224))
        x1 = self.m10(self.input_func(x.narrow(1, 224, 30)))
        return (x0, x1)


# Initializing the model with random weights
m = Model()

# Inputs to the model
input_v = [torch.randn(1, 224), torch.randn(1, 30)]
