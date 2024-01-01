
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64 * 64 * 3 + 128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
 
    def forward(self, input_tensor, other):
        t1 = self.mlp(input_tensor.reshape(input_tensor.size(0), -1))
        t2 = t1 + other
        t3 = torch.relu(t2)
        return t3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 128)
