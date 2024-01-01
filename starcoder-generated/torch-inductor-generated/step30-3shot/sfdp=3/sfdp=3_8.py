
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(12, 16)
 
    def forward(self, x2):
        x = x2 * 0.562341325123
        y = self.dense(x)
        return y

# Initializing the model
model = Model()

# Input that are close to the model's input range
x2 = torch.randn(1, 12)
