
class ModelTanh(torch.nn.Module): 
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.a = torch.tanh
    def forward(self, x1):
        return self.a(x1)
# Inputs to the model
x1 = torch.randn(7, 5, 2, 5)
