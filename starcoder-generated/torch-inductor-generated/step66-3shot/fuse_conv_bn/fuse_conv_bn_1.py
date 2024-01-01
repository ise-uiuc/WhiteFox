
class Model(nn.Module):
    def __init__(self):
        super(EmbeddingBag, self).__init__()
        self.embedding_sum = torch.nn.EmbeddingBag(5, 3, mode='sum', sparse=False)
        self.layer_norm = torch.nn.LayerNorm((5, 3))
    def forward(self, input):
        y = self.embedding_sum(input)
        return self.layer_norm(y)
# Inputs to the model
input = torch.tensor([[1, 1, 2, 4, 0],
                      [4, 3, 1, 2, 5]])
