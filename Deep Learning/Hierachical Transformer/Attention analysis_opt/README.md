## 📊 Attention & Performance Analysis of the Hierarchical Transformer

- 📈  **Evaluate model performance** by analyzing the training and cross-validation <br>
  loss across **every fold, epoch, and batch**. Additionally, examine the loss <br>
  curves during training on the **full dataset** for **every epoch and batch** <br>
  to assess convergence, training stability, and the model's ability to generalize.
- 🎯 **Assess predictive performance** using:
  - True vs. Predicted value plots
  - Bland–Altman plots
  - Residual plots
  - Q–Q plots
  - These visualizations help identify prediction errors, systematic bias, <br>
  heteroscedasticity, and deviations from the assumptions of the model.

- 🧠 **Analyze the model's attention mechanisms** by aggregating attention <br>
  weights across the dataset and visualizing them with:
  - 🔥 Attention heatmaps
  - 🕸️ Attention graphs
  - 📊 Bar charts and scatter plots for average attetion
  -  These visualizations highlight the **attention distribution** and  <br>
  provide insights into which features or input regions the Hierarchical Transformer  <br>
  focuses.

- 🔍 The performance evaluation and attention analysis uses the output data saved in the <br>
  following output directories:
  - ⚡ Energy prediction: `Deep Learning/Hierachical Transformer/Transformer_opt/model_output_energy`
  - 🔋 Power prediction: `Deep Learning/Hierachical Transformer/Transformer_opt/model_output_power`