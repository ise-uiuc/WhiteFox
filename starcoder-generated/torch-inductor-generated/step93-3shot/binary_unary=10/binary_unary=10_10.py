
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        h = torch.zeros([29])
        for i in range(29):
            h[i] = ((inputs[1]*(i+1))/(inputs[2]*2)) - (inputs[3]/(i+1))
        for i in range(29):
            h[i] = 1 / (1 + math.exp(-h[i]))
        return h

# Initializing the model
m = Model()

# Inputs to the model
inputs = torch.zeros([4])
