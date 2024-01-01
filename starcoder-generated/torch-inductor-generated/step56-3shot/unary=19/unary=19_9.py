
class Model(torch.nn.Module):
    def __init__(self, inputFeatures, hiddenSize, outputCategories):
        super().__init__()
        self.linear1 = torch.nn.Linear(inputFeatures, hiddenSize)
        self.linear2 = torch.nn.Linear(hiddenSize, outputCategories)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = torch.sigmoid(x1)
        x3 = self.linear2(x2)
        return x3

# Initializing the model
hiddenSize = 196
inputFeatures, outputCategories = 3*64*64, 10
m = Model(inputFeatures, hiddenSize, outputCategories)

# Inputs to the model
x = torch.randn(batchSize, inputFeatures)
