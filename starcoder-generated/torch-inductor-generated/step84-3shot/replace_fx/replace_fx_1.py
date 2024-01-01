
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = torch.nn.Embedding(4, 5)
        self.e1 = torch.nn.Embedding(5, 4)
    def forward(self, x):
        x = self.e0(x)  # This operator is not in the list of operations we want to replace.
        return self.e1(x) 
# Inputs to the model
x = torch.tensor([[1,2],[3,4]])
