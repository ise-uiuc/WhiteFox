
model = Model()
print(model)
summary(model, (1, 3, 4, 4))
# Inputs to the model
input_x = torch.randn(1, 3, 4, 4)
