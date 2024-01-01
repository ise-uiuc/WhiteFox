
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.GRU(input_size=1, hidden_size=1, bidirectional=False, num_layers=1, batch_first=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 1, 3)
