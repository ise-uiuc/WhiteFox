
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input1 = torch.randn(2, 2)
        self.model1 = torch.nn.Sequential(
            torch.nn.ReLU(), 
            torch.nn.Linear(2, 2), 
            torch.nn.PReLU(), 
            torch.nn.BatchNorm1d(2), 
            torch.nn.Dropout(0.1), 
            torch.nn.Softmax(),
            torch.nn.Sigmoid(), 
            torch.nn.GELU(),
            torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            torch.nn.Softplus(), 
            torch.nn.Tanh()
        )
    def forward(self):
        x1 = self.input1
        x2 = self.model1(x1)
        return x1, x2
# Inputs to the model
x1, x2 = Model()()
