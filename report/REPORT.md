# Deepfake Face Detection with Multimodal Forensic Cues

## 1. Introduction
Deepfake face manipulation has become increasingly realistic and difficult to detect, creating challenges for trust, security, and media authenticity. In this project, we study a multimodal deepfake detector that combines semantic RGB features with explicit forensic cues from the frequency and noise domains. In addition to classification, we also analyze weak localization using Grad-CAM and study cross-dataset generalization.

The main objective of this work is to build a strong detector, understand which architectural components actually help, and evaluate how well the model transfers across datasets such as FF++, Celeb-DF, and DFDC.

## 2. Methodology

### 2.1 Overall Architecture
The proposed detector consists of:
- RGB semantic feature extraction using a CLIP-based visual backbone
- Wavelet frequency feature extraction
- SRM-based noise residual feature extraction
- Gated multimodal fusion
- Multiscale token construction
- Prompt-guided bridge module
- Classification head
- Weak localization using Grad-CAM

### 2.2 Multimodal Forgery Representation
The detector uses three complementary views of the input face:
1. **RGB branch** for semantic facial appearance
2. **Wavelet branch** for frequency-domain artifacts
3. **SRM branch** for residual-noise inconsistencies

These three branches are projected into a common embedding space and fused using a gated fusion module.

### 2.3 Bridge and Classification
After feature fusion, the representation is processed by a prompt-guided bridge and then passed to the classification head for real/fake prediction.

### 2.4 Localization
For interpretability, Grad-CAM is used as a weak evidence-region localization method. This does not produce a precise segmentation mask, but highlights suspicious facial regions that influence the detector decision.

## 3. Experimental Setup

### 3.1 Datasets
- **FF++**: main in-domain training and evaluation dataset
- **Celeb-DF**: external cross-dataset benchmark
- **DFDC**: external cross-dataset benchmark
- **DeepFakeFace**: diffusion-based fake face dataset used in auxiliary experiments

### 3.2 Metrics
The following metrics are reported:
- Accuracy
- F1-score
- Precision
- Recall
- AUC
- EER

For Celeb-DF and DFDC, video-level evaluation is used by aggregating frame-level predictions.

## 4. Main Detector Performance on FF++

The final FF++ detector uses:
- wavelet branch enabled
- SRM branch enabled
- bridge enabled
- reasoning transformer removed
- contrastive loss enabled

### Final FF++ Test Result
| Metric | Value |
|---|---:|
| Accuracy | 0.9188 |
| F1 | 0.9368 |
| Precision | 0.9284 |
| Recall | 0.9453 |
| AUC | 0.9720 |
| EER | 0.0822 |

## 5. Ablation Study

### 5.1 Ablation Table
| Variant | AUC | EER | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Full | 0.9689 | 0.0891 | 0.8944 | 0.9119 |
| No SRM | 0.9571 | 0.1034 | 0.8959 | 0.9198 |
| No wavelet | 0.9619 | 0.1011 | 0.9017 | 0.9216 |
| No wavelet + no SRM | 0.9583 | 0.1104 | 0.8975 | 0.9196 |
| No bridge | 0.9690 | 0.0864 | 0.9133 | 0.9306 |
| No contrastive | 0.9588 | 0.1034 | 0.8946 | 0.9149 |
| No reasoning | 0.9715 | 0.0822 | 0.9207 | 0.9377 |

### 5.2 Discussion
The ablation study shows that:
- the **SRM branch** improves performance
- the **wavelet branch** also improves performance
- **contrastive loss** is useful
- the **reasoning transformer** does not help in the final FF++ setting
- the best final detector is obtained by removing the reasoning transformer while keeping the bridge

## 6. Cross-Dataset Generalization

### 6.1 Single-Source Cross-Dataset Results
| Train | Test | AUC | EER |
|---|---|---:|---:|
| FF++ | FF++ | 0.9720 | 0.0822 |
| FF++ | Celeb-DF | 0.7852 | 0.2657 |
| FF++ | DFDC | 0.6349 | 0.4114 |
| Celeb-DF | Celeb-DF | 1.0000 | 0.0000 |
| Celeb-DF | FF++ | 0.6392 | 0.4036 |
| Celeb-DF | DFDC | 0.6367 | 0.3954 |

### 6.2 Multi-Source Training
| Train | Test | AUC | EER |
|---|---|---:|---:|
| FF++ + Celeb-DF | DFDC | 0.6941 | 0.3644 |
| FF++ + Celeb-DF + DFF | DFDC | 0.6945 | 0.3654 |
| FF++ + Celeb-DF + DG-Aug | DFDC | 0.6609 | 0.3943 |

### 6.3 Discussion
The cross-dataset results show a substantial domain gap. Single-source training performs well in-domain but degrades strongly on unseen datasets. Multi-source training on FF++ and Celeb-DF improves DFDC transfer, while adding the diffusion dataset or stronger augmentation does not provide further gains.

## 7. Localization and Explanation

### 7.1 Localization
Grad-CAM is used to highlight weak evidence regions in the face. The visualizations show that the detector often focuses on suspicious facial regions in fake images and central facial structure in real images.

### 7.2 Explanation
A constrained evidence-based explanation strategy is used for final presentation. The explanation is derived from the detector decision and highlighted evidence region, rather than relying on a fully generative explanation model.

### 7.3 Example Explanation Templates
- **Fake**: The image is predicted as fake because the highlighted facial region shows suspicious local inconsistencies in texture or facial structure.
- **Real**: The image is predicted as real because the highlighted facial region appears visually consistent, and no strong manipulation artifacts are evident.

## 8. Limitations
The detector remains strong in-domain but does not generalize robustly across all external datasets. Cross-dataset transfer remains challenging, especially for DFDC. The current localization is weakly interpretable rather than mask-level precise, and the learned explanation module remains preliminary.

## 9. Conclusion
This project presents a multimodal deepfake detector combining RGB, wavelet, and SRM cues. The final detector achieves strong in-domain performance on FF++, and the ablation study shows that forensic branches and contrastive learning are useful while the reasoning transformer is unnecessary in the best configuration. Cross-dataset experiments reveal substantial domain shift, though multi-source training on FF++ and Celeb-DF improves transfer to DFDC. Overall, the work demonstrates a strong detector, useful weak localization, and a clear analysis of generalization behavior.
