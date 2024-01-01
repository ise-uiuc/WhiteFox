
class Model(torch.nn.Module):
    def forward(self, input, input_data):
        x = torch.relu(input)
        y = torch.relu(input_data)
        z = torch.relu(x) + y
        return z
# Inputs to the model
input = torch.randn(100, 100)
input_data = torch.randn(100, 100)
