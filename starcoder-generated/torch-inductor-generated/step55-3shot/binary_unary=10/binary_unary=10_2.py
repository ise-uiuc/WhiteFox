
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        feature_num = 128
        self.linear1 = torch.nn.Sequential(
                torch.nn.Linear(1024, feature_num),
                torch.nn.ReLU(inplace=True),
        )
        self.linear2 = torch.nn.Linear(feature_num, 10)

    def forward(self, x1):
        return self.linear2(self.linear1(x1))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 1024)
