# NeRFCom

Official PyTorch implementation for **"NeRFCom: Feature Transform Coding Meets Neural Radiance Field for Free-View 3D Scene Semantic Transmission"**.


## Training

As described in the paper, NeRFCom training is divided into three stages:

+ Pretrain the NeRF module: You can either train from scratch or use publicly available pretrained models. We recommend using models with geometric priors, such as [DVGO](https://github.com/sunset1995/DirectVoxGO), [TensoRF](https://github.com/apchenstu/TensoRF), or [K-Planes](https://github.com/sarafridov/K-Planes).

+ Train the Encoder and Decoder modules: We recommend performing training in two sub-stages (w/o and w/ channel) to stabilize convergence and improve performance.

+ Joint training with all components: After the above two stages, jointly fine-tune the entire NeRFCom pipeline for best performance.

> **Maybe Help:**
> + The training mode for each stage currently needs to be manually adjusted in the code.  
> + You are encouraged to monitor the learning rate schedules and gradient flows to verify that only the intended modules are updated during each phase.
> + This repository is currently undergoing reorganization. Partial code has been uploaded, and more updates will be released soon.

## Acknowledgements

This project builds upon the excellent work of the following repositories:

- [DVGO](https://github.com/sunset1995/DirectVoxGO)
- [K-Planes](https://github.com/sarafridov/K-Planes)
- [NTSCC](https://github.com/wsxtyrdd/NTSCC_JSAC22)  

We sincerely thank the authors of these projects for their contributions.

---
