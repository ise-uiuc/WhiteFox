
class Model():
    def forward(self, x):
        v1 = torch.flatten(self.linear1(x), 1)
        v2 = torch.tanh(v1)
        v3 = torch.tanh(self.linear2(v2))
        v4 = v3.reshape(1, 3, 64, 64)
        return v4

# Initializing the model
self.linear1 = nn.Linear(3 * 64 * 64, 256)
self.linear2 = nn.Linear(256, 3 * 64 * 64)

# Inputs to the model
x = torch.rand(1, 3, 64, 64)
