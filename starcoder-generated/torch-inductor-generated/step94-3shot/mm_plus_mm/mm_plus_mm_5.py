
class Model(pytorch_model.PyTorchModel):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(3, 5)

    def forward(self, x, y):
        x = self.linear1(x)
        return x.sum() + y.sum()
# Inputs to the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(2, 3, device=device)
y = torch.randn(2, 3, device=device)
