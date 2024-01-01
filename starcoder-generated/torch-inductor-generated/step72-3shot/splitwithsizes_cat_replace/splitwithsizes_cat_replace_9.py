
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.EmbeddingBag(3, 3, mode="sum")
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.LongTensor([[0, 1, 2, 3], [3, 4, 0, 0]])
