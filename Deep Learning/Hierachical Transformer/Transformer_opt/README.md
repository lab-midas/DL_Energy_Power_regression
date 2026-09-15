# 🧠 Hierarchical Multimodal Transformer

Custom transformer architecture designed for energy prediction from MRI <br>
sequence data with interpretable attention mechanisms.

---

## 🎯 Overview
- 🔀 **Multimodal Processing:** Dedicated processing channels for Numerical <br>
and Text+Categorical data
- 🏗️ **Hierarchical Design:** Progressive dimensionality reduction through <br>
bottleneck architecture for improved generalization
- 🔗 **Cross-Modal Interaction:** Channels exchange information and learn <br>
joint representations during processing
- 🧩 **Unified Prediction:** Final representations from both channels are <br>
combined for energy consumption prediction
- 🧐 **Interpretable Analysis:** Direct access to attention weights enables <br>
comprehensive feature importance evaluation

---

## 📋 Implementation

**📄 transformer.ipynb**

## 🏗️ Architecture Details
![alt text](image-1.png)

### 🛤️ **Dual-Channel Processing**
- **🔢 Numerical Channel:** Input Projection Layer (Dense Layer) processes <br>
numerical MRI parameters
- **🔤 Text+Categorical Channel:** Embedding Layer (Look-up Matrix) handles <br>
textual and categorical features

### 🧠 **Hierarchical Encoder-Decoder Structure**
- **🔄 Encoder Blocks:** Utilize both self-attention and cross-attention <br>
mechanisms for intra- and inter-modal learning
- **🎯 Decoder Blocks:** Use only self-attention mechanisms to process <br>
concatenated representations
- **📉 Progressive Reduction:** Model dimensions are systematically reduced <br>
across Decoder blocks
- **➡️ Projection Layers:** Dense layers between blocks enable controlled <br>
dimensionality reduction
- **🔗 Channel Fusion:** Encoded latent sequences from both channels are <br>
concatenated for unified representation

### 🔀 **Cross-Attention in Encoder Blocks**
- 🔄 **Two-Stage Processing:** Encoder blocks are split into two sequential <br>
stages for proper cross-modal interaction
- 1️⃣ **Stage 1 - Self-Attention:** Each modality (numerical and text+categorical) <br>
first processes its own features independently through self-attention layers
- 2️⃣ **Stage 2 - Cross-Attention:** After self-attention, both modalities exchange <br>
information through cross-attention mechanisms
- ⚙️ **Why This Split?** Cross-attention requires the refined representations <br>
from self-attention as input—each channel must first understand its own features <br>
before learning relationships across modalities
- 🔗 **Information Flow:** Numerical features attend to text+categorical features, <br>
and vice versa, enabling bidirectional cross-modal learning

### 🎯 **Output Generation**
- **🎯 Dense Regression Layer:** A fully connected layer maps the flattened <br>
features to energy consumption predictions
- **📊 Attention Export:** All attention weights are directly accessible for <br>
comprehensive interpretability analysis

This architecture combines the strengths of multimodal processing with <br>
hierarchical representation learning, enabling both accurate predictions and <br>
meaningful insights into MRI parameter relationships.