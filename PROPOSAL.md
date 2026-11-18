# CSE 422 Project Proposal
## Title
Robust PCA for Video Background Subtraction
## Team Members
Alexander Tarnavski (alextarn@uw.edu)
Pak Ho Lam (phlamah@uw.edu)


## Reference Paper
1. Candes, E. J., Li, X., Ma, Y., & Wright, J. (2009, December 18). Robust principal component analysis? arXiv.Org. https://doi.org/10.48550/arXiv.0912.3599

2. Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices (1009.5055v3). arXiv. https://doi.org/10.48550/arXiv.1009.5055

## Aim
To develop an efficient video background removal system by leveraging Robust Principal Component Analysis (Robust PCA) and optimized low-rank matrix multiplication techniques. The system aims to separate static background information from moving foreground objects in an input video while improving computational efficiency and maintaining robust background estimation.

## Plan
The project will make use of common Python libraries, including OpenCV for video processing, frame extraction, resizing, and conversion into matrices suitable for linear algebra operations. NumPy and SciPy will be used to implement the Robust PCA algorithm and perform the necessary matrix decomposition and optimization operations. To improve computational efficiency, different low-rank matrix multiplication and decomposition techniques, such as truncated SVD and randomized SVD, will be explored and compared against a baseline implementation. Matplotlib will be used to visualize the original video, extracted background, and foreground components under different algorithm parameters. Finally, the performance of different implementations will be evaluated in terms of background removal quality, processing time, and memory consumption.

## Extension
This project extends beyond baseline reproduction by evaluating the effect of key hyperparameters and conducting comprehensive benchmarking. Specifically, we will perform a sensitivity analysis on the regularization parameter $\lambda$ to examine its trade-off between sparse noise removal and background retention. Additionally, we will assess the algorithm's robustness across diverse video scenes with varying lighting and motion characteristics.

## Dataset
We will make use of a self recorded video, a public surveillance benchmark clip from PETS dataset and UW Panopto lecture video (if applicable). To keep operation efficient, videos will be downsampled and tranform to greyscale.

## Compute
All experiments will be conducted locally on standard CPU hardware using Python and its scientific computing libraries, eliminating the need for specialized external or cloud GPU acceleration. Note that this computational scope may be revisited if we extend our implementation to full-color/larger video processing.