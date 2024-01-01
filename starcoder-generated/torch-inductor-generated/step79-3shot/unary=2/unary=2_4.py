
model = torch.nn.Sequential(
            torch.nn.Linear(9, 16, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(16, 9),
        )
# Inputs to the model
x1 = torch.randn(7, 9)
