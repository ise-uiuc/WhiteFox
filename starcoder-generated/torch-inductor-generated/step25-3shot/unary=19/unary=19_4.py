
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1536, 10)
 
    def forward(self, x1):
        x1 = nn.Linear(in_features, out_features)(x1)
        x1 = torch.sigmoid(x1)
        x1 = torch.softmax(x1, dim=0)
        return x1

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(5, 1536)
