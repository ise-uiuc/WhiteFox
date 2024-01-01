
torch.manual_seed(0)  
num_features = 100
num_classes = 200
torch.manual_seed(1)

# Inputs to the model
x2 = torch.randn(5, num_features)
m = torch.nn.Linear(num_features, num_classes, False)
m = torch.nn.Sigmoid()
