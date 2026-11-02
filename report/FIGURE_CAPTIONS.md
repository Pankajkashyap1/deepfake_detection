# Figure Captions and Report Notes

## Figure 1. Localization examples using the final detector
The figure shows weak evidence-region localization obtained using the localization head of the final FF++ detector. For each example, the original face image, normalized heatmap, and overlay are shown. In fake images, the model highlights suspicious facial regions that may contain local inconsistencies, while in real images the response is generally more consistent and less focused on suspicious artifacts.

### Suggested subfigure captions
- **tp_fake_1.png**: Correctly detected fake image with highlighted suspicious facial region.
- **tp_fake_2.png**: Another correctly detected fake sample showing concentrated response over a manipulated facial area.
- **tn_real_1.png**: Correctly detected real image with relatively consistent facial evidence.
- **tn_real_2.png**: Correctly detected real image with no strong suspicious region.

## Figure 2. Failure case analysis
If false-positive and false-negative examples are available, they can be described as follows:
- **fp_1.png**: False positive case where the detector incorrectly identifies a real image as fake, indicating remaining domain bias or sensitivity to unusual facial appearance.
- **fn_1.png**: False negative case where the detector fails to identify a fake image, suggesting that some manipulations remain subtle or visually consistent.

## Main result statement
The final detector achieves strong in-domain performance on FF++, with:
- Accuracy = 0.9188
- F1 = 0.9368
- AUC = 0.9720
- EER = 0.0822

## Main ablation statement
The ablation study shows that:
- SRM and wavelet forensic branches improve performance
- contrastive supervision is useful
- removing the reasoning transformer gives the best final detector
- the prompt-guided bridge remains beneficial

## Main cross-dataset statement
Cross-dataset evaluation reveals substantial domain shift:
- FF++ -> Celeb-DF: AUC 0.7852, EER 0.2657
- FF++ -> DFDC: AUC 0.6349, EER 0.4114

Multi-source training improves transfer:
- FF++ + Celeb-DF -> DFDC: AUC 0.6941, EER 0.3644

Additional experiments show that:
- adding diffusion-based data does not significantly improve DFDC transfer
- strong domain-generalization augmentation also does not improve DFDC in the current setup

## Explanation module statement
An explanation module based on detector evidence was explored as part of the project objective. Since the fully generative explanation pipeline remained preliminary and not fully reliable under all cases, the final submission uses detector-consistent evidence-based explanations for safe presentation.

