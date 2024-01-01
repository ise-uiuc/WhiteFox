
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 50, 5)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.pool(F.relu(x2))
        x4 = self.conv2(x3)
        x5 = self.pool(F.relu(x4))
        x6 = x5.view(-1, 4*4*50)
        x7 = self.fc1(x6)
        x8 = self.drop(F.relu(x7))
        x9 = self.fc2(x8)
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
model = Model()
gm = TorchModel(model)
gm.graph.graph.lint()
for node in gm.graph.find_module_nodes(aten.dropout):
    gm.graph.erase_node(node)
print(gm)
