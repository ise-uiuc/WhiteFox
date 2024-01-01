
class model(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(model, self).__init__()
        self.layer1 = torch.nn.Linear(2, hidden_size)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        z1 = self.layer1(input)
        a1 = z1
        z2 = self.relu(a1)
        y1 = self.sigmoid(self.layer2(z2))
        return y1
# Inputs to the model
inputs = torch.randn(2)
