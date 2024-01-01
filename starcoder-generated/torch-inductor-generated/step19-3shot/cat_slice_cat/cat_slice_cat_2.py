
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *args):
        lstm = torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        lstm1, _ = lstm(args[0])
 
        v1 = torch.cat([args[0], lstm1], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:16]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 50, 64)
