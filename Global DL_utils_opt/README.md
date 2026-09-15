# 📉 Loss Collection & Visualization

This module provides utility classes and functions for collecting, organizing, <br>
and visualizing loss values during the training of the Transformer model. <br>
Losses are stored hierarchically, allowing detailed analysis of the training <br>
process from individual batches to complete cross-validation folds.

**Key Features:**
- 🗂️ Hierarchical loss management: Stores losses using the structure <br>
  Fold → Epoch → Batch for detailed tracking of the training process
- 🏋️ Training and 🧪 validation loss tracking for cross-validation and full-data training
- 🔄 Cross-validation loss analysis and comparison across individual folds
- 📈 Epoch- and batch-level loss visualization
- 📊 Mean loss curves across cross-validation folds
- 🔍 Visualization of individual fold learning paths alongside mean loss curves
- 🚧 Visualization of epoch boundaries and validation intervals
- 🧮 Automatic handling of different numbers of epochs or batches across folds
- 🎨 Interactive Plotly-based visualizations with customizable lines and markers
- ⚙️ Flexible plotting for different training types and aggregation levels
- 💾 Optional saving of generated figures in multiple formats
- 🎯 Monitoring of model convergence, training stability, and potential overfitting
