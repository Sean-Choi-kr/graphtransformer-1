import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.graph_transformer_edge_layer import GraphTransformerLayer

class GraphTransformerNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim,           # Initial input feature dimension
        hidden_dim,       # Hidden dimension
        out_dim,         # Output dimension (number of classes)
        num_layers,      # Number of GraphTransformer layers
        num_heads,       # Number of attention heads
        dropout=0.1,     # Dropout rate
        residual=True    # Whether to use residual connections
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.graph_transformer_layers = nn.ModuleList()
        
        # First layer (input dimension -> hidden dimension)
        self.graph_transformer_layers.append(
            GraphTransformerLayer(
                in_dim=in_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual=residual
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.graph_transformer_layers.append(
                GraphTransformerLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    residual=residual
                )
            )
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, h, e):
        # Apply GraphTransformer layers
        for i in range(self.num_layers):
            h, e = self.graph_transformer_layers[i](g, h, e)
            
        # Apply final classifier
        logits = self.classifier(h)
        
        return logits