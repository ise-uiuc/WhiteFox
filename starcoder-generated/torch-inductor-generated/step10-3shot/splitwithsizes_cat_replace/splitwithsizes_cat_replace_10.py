
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.split = torch.nn.LSTM(input_size=32, hidden_size=3, num_layers=3, batch_first=False)
        self.cat = torch.nn.Sequential(torch.nn.MaxPool2d(3, 2, 1, 1), torch.nn.MaxPool2d(5, 4, 2, 2), torch.nn.MaxPool2d(3, 1, 1, 0))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = self.split(v1)[0]
        concatenated_tensor = self.cat((concatenated_tensor, v1))
        return (concatenated_tensor, v1, split_tensors)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
