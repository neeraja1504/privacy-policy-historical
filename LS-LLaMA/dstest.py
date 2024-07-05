import torch
import torch.nn as nn
import deepspeed

# Define a simple feedforward neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load DeepSpeed configuration from file
deepspeed_config = '/scratch/bbzy/neeraja1504/CI/LS-LLaMA/ds_config.json'

# Initialize DeepSpeed
model = SimpleModel()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=deepspeed_config,
    model_parameters=model.parameters(),
    training_data=None,  # Not needed for this simple example
    lr_scheduler=None    # No learning rate scheduler for simplicity
)

# Generate some synthetic data
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randn(100, 1)   # 100 targets

# Training loop
for epoch in range(10):  # 10 epochs
    # Forward pass
    outputs = model_engine(X)
    loss = nn.functional.mse_loss(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    model_engine.backward(loss)
    optimizer.step()

    # Print progress
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

print("Training completed!")
