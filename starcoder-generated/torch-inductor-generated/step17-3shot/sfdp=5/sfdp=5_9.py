
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.dim = 32
    def forward(self, query):
        output = torch.tanh(query)
        return output
# Inputs to the model
input_ids = randint(low=0, high=5, size=(1, 256))
input_ids = torch.nn.functional.one_hot(input_ids).long() * 3
