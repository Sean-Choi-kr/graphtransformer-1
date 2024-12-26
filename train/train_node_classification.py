# Initialize model
import torch
from nets.node_classification import GraphTransformerNodeClassifier
import torch.nn as nn

input_feature_dim = 1
num_classes = 2

model = GraphTransformerNodeClassifier(
    in_dim=input_feature_dim,      # Your input feature dimension
    hidden_dim=64,                 # Hidden dimension size
    out_dim=num_classes,           # Number of classes
    num_layers=2,                  # Number of GraphTransformer layers
    num_heads=4,                   # Number of attention heads
    dropout=0.1,
    residual=True
)

# Example training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(g, features, edge_features, labels):
    model.train()
    
    # Forward pass
    logits = model(g, features, edge_features)
    loss = criterion(logits, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example prediction
def predict(g, features, edge_features):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, edge_features)
        predictions = torch.argmax(logits, dim=1)
    return predictions