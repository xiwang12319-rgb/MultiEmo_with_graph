import time
import torch


def main():
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    hip_version = torch.version.hip
    cuda_version = torch.version.cuda

    device = torch.device("cuda" if cuda_available else "cpu")

    print("torch.__version__:", torch_version)
    print("torch.cuda.is_available():", cuda_available)
    print("torch.version.hip:", hip_version)
    print("torch.version.cuda:", cuda_version)
    print("Selected device:", device)

    x = torch.randn(64, 64, device=device)
    y = torch.randn(64, 64, device=device)
    print("Tensor device:", x.device)

    start = time.time()
    z = x @ y
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed_ms = (time.time() - start) * 1000.0
    print("Matmul result shape:", z.shape)
    print("Matmul time (ms):", round(elapsed_ms, 4))

    if cuda_available:
        print("Conclusion: CUDA available")
    elif hip_version is not None:
        print("Conclusion: DCU backend detected")
    else:
        print("Conclusion: CPU only")
        if cuda_version is None and hip_version is None:
            print("Conclusion: accelerator not enabled in current torch")


if __name__ == "__main__":
    main()
