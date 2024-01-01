
class Model(torch.nn.Module):
    def __init__(self, input_size=64, output_size=256, hidden_size=128):
        super().__init__()
        self.bn0 = torch.nn.BatchNorm2d(input_size)
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.linear2 = torch.nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, inputs):
        x = self.bn0(inputs)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
