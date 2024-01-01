
from collections import namedtuple
class Model(torch.nn.Module):
    def __init__(self, linear1_weight, linear1_bias, linear2_weight, linear2_bias):
        super().__init__()
        Linear = namedtuple('Linear', 'weight bias')
        self.linear1 = Linear(torch.from_numpy(linear1_weight), torch.from_numpy(linear1_bias))
        self.linear2 = Linear(torch.from_numpy(linear2_weight), torch.from_numpy(linear2_bias))

    def forward(self, x):
        v1 = torch.addmm(self.linear1.bias, x, self.linear1.weight.t())
        v2 = torch.relu(v1)
        v3 = torch.addmm(self.linear2.bias, v2, self.linear2.weight.t())
        return v3

# Initializing the model
linear1_weight = np.random.rand(5, 3).astype(np.float32)
linear1_bias = np.random.rand(5).astype(np.float32)
linear2_weight = np.random.rand(5, 5).astype(np.float32)
linear2_bias = np.array([-1.3, 0.2, 0.6, -0.5, 0.5]).astype(np.float32)
m = Model(linear1_weight, linear1_bias, linear2_weight, linear2_bias)

# Inputs to the model
x = torch.randn(1, 3)
