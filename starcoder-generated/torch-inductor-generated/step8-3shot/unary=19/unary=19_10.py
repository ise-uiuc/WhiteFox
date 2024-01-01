
class Model(torch.nn.Module):
    def __init__(self, input_channel = 3, output_channel = 1, num_hidden_nodes = 32):
        super(Model, self).__init__()
        self.num_hidden_nodes = num_hidden_nodes
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(input_channel * 32 * 32, self.num_hidden_nodes)
        self.linear2 = torch.nn.Linear(self.num_hidden_nodes, self.num_hidden_nodes)
        self.linear3 = torch.nn.Linear(self.num_hidden_nodes, self.num_hidden_nodes)
        self.output_layer = torch.nn.Linear(self.num_hidden_nodes, output_channel)
 
    def forward(self, x1):
        v1 = self.flatten(x1)
        v2 = self.linear1(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.linear2(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.linear3(v5)
        v7 = torch.sigmoid(v6)
        v8 = self.output_layer(v7)
        result = torch.sigmoid(v8)
        return result
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
