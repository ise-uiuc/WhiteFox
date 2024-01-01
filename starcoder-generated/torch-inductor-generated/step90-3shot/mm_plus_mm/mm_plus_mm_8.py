
class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(896, 512)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(100, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x)
        x123 = x1 + x2 + x3
        return self.classifier(self.relu(x123))

model = Model(10)
# Input to the model
x = torch.randn(1, 896)
# Model end
