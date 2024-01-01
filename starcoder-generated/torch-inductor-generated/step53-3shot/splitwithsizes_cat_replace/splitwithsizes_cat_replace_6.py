
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ReLU()
    def forward(self, v1, v2):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat([split_tensors[i] for i in range(3)], dim=1) # Replace the specific indices with i in [0,3) to make the pattern more general
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1), torch.squeeze(v2))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
