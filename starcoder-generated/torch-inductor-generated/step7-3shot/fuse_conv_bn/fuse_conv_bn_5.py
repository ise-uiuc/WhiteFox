
device = torch.device('cuda:0')
torch.manual_seed(123)
model = Model()
# Inputs to the model
x = torch.randn(5, 224, 224, device=device)
