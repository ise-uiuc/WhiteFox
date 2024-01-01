
class Model2(torch.nn.Module):
    def __init__(self, size):
        super(Model2, self).__init__()
        self.size = size
 
    def forward(self, x1):
        concated = torch.cat([x1, x1+1, x1+2, x1+3], dim=1)
        sliced = concated[:, :self.size]
        second_concated = torch.cat([concated[:, :self.size], concated[:, self.size:]], dim=1)

        return second_concated

# Initializing the model
model2 = Model2(10)

# Inputs to the model
x1 = torch.randn(1, 20, 20, 20)
