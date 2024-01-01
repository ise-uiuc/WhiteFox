
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 784)
prediction = m(x1)

# Check the prediction
assert (
    isinstance(prediction, torch.Tensor) and prediction.size() == torch.Size([1, 10])
)
print('Success!')
