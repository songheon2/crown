"""
Custom 바이너리 형식 → LiRPA FullyConnectedNetwork 변환기

Custom 바이너리 형식 (OnnxToCustom.py / CustomBinary.py와 동일, 리틀엔디안 고정):
    magic   : 4 bytes   b'NNCB'
    version : int32      (=1)
    m       : int32      (가중치 레이어 수)
    sizes   : int32[m+1] (레이어별 노드 수 n_0..n_m)
    이후 m개 블록:
        W_i : float64[n_out, n_in]  (row-major)
        b_i : float64[n_out]

활성화 함수는 파일에 저장되어 있지 않으므로, 기존 관례(GenericNNEncoding.py,
OnnxToCustom.py)를 따라 은닉층은 ReLU, 출력층은 항등(linear, logit 그대로)으로
고정한다.
"""

from __future__ import annotations

import struct

import numpy as np

from lirpa_forward_backward_fc import FullyConnectedNetwork, LiRPAForward, LiRPABackward

_MAGIC = b"NNCB"
_VERSION = 1


def load_custom_network(filepath: str) -> FullyConnectedNetwork:
    """Custom 바이너리 형식(.bin) 파일을 읽어 FullyConnectedNetwork로 반환한다."""
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise ValueError(f"'{filepath}'는 유효한 커스텀 바이너리 파일이 아닙니다 (magic 불일치)")

        version, = struct.unpack("<i", f.read(4))
        if version != _VERSION:
            raise ValueError(f"지원하지 않는 커스텀 바이너리 버전입니다: {version}")

        m, = struct.unpack("<i", f.read(4))
        sizes = list(struct.unpack(f"<{m + 1}i", f.read(4 * (m + 1))))

        weights = []
        biases = []
        for i in range(m):
            n_in, n_out = sizes[i], sizes[i + 1]
            W = np.fromfile(f, dtype="<f8", count=n_out * n_in).reshape(n_out, n_in)
            b = np.fromfile(f, dtype="<f8", count=n_out)
            weights.append(W)
            biases.append(b)

    activations = ["relu"] * (m - 1) + ["linear"]

    return FullyConnectedNetwork(weights=weights, biases=biases, activations=activations)


def run_xor_demo(filepath: str, eps: float = 0.02) -> None:
    """
    Custom 형식으로 로드한 XOR 네트워크에 대해 LiRPA forward/backward bound를
    실행한다. 출력층이 linear(logit)이므로, sigmoid(logit) > 0.5 는 logit > 0
    으로, sigmoid(logit) < 0.5 는 logit < 0 으로 판정한다.
    """
    network = load_custom_network(filepath)
    forward_verifier = LiRPAForward()
    backward_verifier = LiRPABackward(forward_verifier)

    points = [
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
    ]

    print(f"Custom 네트워크({filepath})에 대한 LiRPA forward/backward 검증")
    print(f"Perturbation: L_inf epsilon = {eps}")
    print()

    for x0 in points:
        logit = network.forward(x0)[0]
        _, fwd_lb, fwd_ub, _ = forward_verifier.bound(network, x0, eps)
        _, bwd_lb, bwd_ub, _ = backward_verifier.bound(network, x0, eps)
        expected = int(round(float(x0[0]))) ^ int(round(float(x0[1])))

        if expected == 1:
            fwd_certified = bool(fwd_lb[0] > 0.0)
            bwd_certified = bool(bwd_lb[0] > 0.0)
            condition = "lower bound > 0"
        else:
            fwd_certified = bool(fwd_ub[0] < 0.0)
            bwd_certified = bool(bwd_ub[0] < 0.0)
            condition = "upper bound < 0"

        print(f"x0={x0.tolist()}, expected={expected}, logit={logit:.6f}")
        print(f"  forward  bound=[{fwd_lb[0]:.6f}, {fwd_ub[0]:.6f}], certified={fwd_certified} ({condition})")
        print(f"  backward bound=[{bwd_lb[0]:.6f}, {bwd_ub[0]:.6f}], certified={bwd_certified} ({condition})")


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "Custom/xor_network.bin"
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.22
    run_xor_demo(filepath, eps)
