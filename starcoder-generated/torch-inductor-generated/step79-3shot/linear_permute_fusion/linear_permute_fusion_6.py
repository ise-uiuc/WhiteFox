
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_bag = torch.nn.EmbeddingBag(4, 4)
    def forward(self, i2):
        return self.embedding_bag.permute(1, 0)
# Inputs to the model
i2 = torch.tensor([2])
