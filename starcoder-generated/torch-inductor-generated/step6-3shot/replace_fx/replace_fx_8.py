
class Network(nn.Module):
    def __init__(self, layer_size, input_size, output_size):
        super().__init__()
        self.fc0 = nn.Linear(input_size, layer_size)
        self.fc1 = nn.Linear(layer_size, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x = self.fc0(data)
        x = self.dropout(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.fc1(x)
        return torch.softmax(x, dim=0)

input_size = 3
output_size = 10
layer_size = 100

net = Network(layer_size=layer_size, input_size=input_size, output_size=output_size)
# Inputs to the model
data = torch.randn(1, input_size)

