"""
mMIMO top-k 안테나 선택에 대한 로컬 강건성(local robustness) 검증 모듈.

배경
----
네트워크는 256차원 입력(16x16 H^T H 행렬을 flatten)을 받아 16차원 출력(안테나별
스코어)을 내고, 그 중 스코어가 가장 큰 top-k(k=8)개 안테나를 선택하는 데 쓰인다.

"정답"은 데이터셋 라벨이 아니라 원본(섭동 없는) 입력 x0에 대해 네트워크 자신이
고른 top-k 선택 결과다. 이 모듈은 L_inf eps 반경 안의 모든 섭동에 대해 그 top-k
선택 결과가 절대 바뀌지 않음을 CROWN(backward LiRPA) bound로 증명한다.

Certified 판정 알고리즘
-----------------------
1. backward LiRPA로 16개 출력 각각의 하한(lb)/상한(ub)을 구한다.
2. lb 기준 내림차순 정렬값을 sorted_lb, ub 기준 내림차순 정렬값을 sorted_ub라 하면,
   sorted_lb[k-1] > sorted_ub[k]  (k번째로 큰 lb > (k+1)번째로 큰 ub)
   가 성립해야 한다. 이게 성립하면, 어떤 섭동이 오더라도(각 출력이 각자의 lb~ub
   범위 안 어디를 찍든) lb 기준 top-k 그룹의 값 구간과 나머지 그룹의 값 구간이
   절대 겹치지 않으므로, top-k 선택 결과가 섭동에 무관하게 고정된다는 것이 보장된다.
   (참고: 이 조건은 top-k 집합과 나머지 집합을 미리 분리하지 않고 전체 정렬값만으로
   비교하는 보수적(sufficient) 조건이라 실제보다 다소 엄격할 수 있다.)
3. 조건이 성립하면 lb 기준 상위 k개 인덱스를 "섭동 하에서도 보장되는 선택 결과"로
   삼고, 원본(clean) top-k와 비교해 일치 개수가 k - max_diff 이상이면 certified.
   조건이 성립하지 않으면 바로 미인증(certified=False)이다.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from CustomToLirpa import load_custom_network
from lirpa_forward_backward_fc import Array, FullyConnectedNetwork, LiRPABackward, LiRPAForward


def crown_bounds(network: FullyConnectedNetwork, x0: Array, eps: float) -> tuple[Array, Array]:
    """backward LiRPA(CROWN)로 모든 출력의 (lower, upper) bound를 구한다."""
    backward_verifier = LiRPABackward(LiRPAForward())
    _, lb, ub, _ = backward_verifier.bound(network, x0, eps)
    return lb, ub


def certify_topk_selection(
    y0: Array,
    lb: Array,
    ub: Array,
    k: int = 8,
    max_diff: int = 0,
) -> Dict:
    """
    원본 출력 y0과 CROWN bound (lb, ub)로부터 top-k 선택의 로컬 강건성을 판정한다.
    """
    clean_topk = set(np.argsort(y0)[-k:].tolist())

    order_lb = np.argsort(lb)[::-1]
    sorted_lb = lb[order_lb]
    sorted_ub = np.sort(ub)[::-1]

    gap_ok = bool(sorted_lb[k - 1] > sorted_ub[k])
    if not gap_ok:
        return {
            "certified": False,
            "gap_ok": False,
            "clean_topk": sorted(clean_topk),
            "guaranteed_topk": None,
            "match_count": None,
        }

    guaranteed_topk = set(order_lb[:k].tolist())
    match_count = len(guaranteed_topk & clean_topk)
    certified = match_count >= (k - max_diff)

    return {
        "certified": certified,
        "gap_ok": True,
        "clean_topk": sorted(clean_topk),
        "guaranteed_topk": sorted(guaranteed_topk),
        "match_count": match_count,
    }


def verify_point(
    network: FullyConnectedNetwork,
    x0: Array,
    eps: float,
    k: int = 8,
    max_diff: int = 0,
) -> Dict:
    """단일 입력점 x0에 대해 top-k 선택의 로컬 강건성을 검증한다."""
    x0 = np.asarray(x0, dtype=float)
    y0 = network.forward(x0)
    lb, ub = crown_bounds(network, x0, eps)
    return certify_topk_selection(y0, lb, ub, k=k, max_diff=max_diff)


def sweep(
    network: FullyConnectedNetwork,
    X: Array,
    eps_list: List[float],
    k: int = 8,
    max_diff: int = 0,
) -> Dict[float, Dict]:
    """
    여러 입력점(X: (N, input_dim))과 여러 eps 값에 대해 top-k 선택 강건성 검증을
    반복하고, eps별 certified(T)/미인증(F) 개수와 비율을 반환한다.
    """
    X = np.asarray(X, dtype=float)
    results: Dict[float, Dict] = {}

    for eps in eps_list:
        t_count = 0
        f_count = 0
        for i in range(X.shape[0]):
            result = verify_point(network, X[i], eps, k=k, max_diff=max_diff)
            if result["certified"]:
                t_count += 1
            else:
                f_count += 1
            if (i + 1) % 1000 == 0:
                print(f"  eps={eps:>10}: {i + 1}/{X.shape[0]} points done (T={t_count} F={f_count})")
        total = t_count + f_count
        results[eps] = {
            "T": t_count,
            "F": f_count,
            "ratio": t_count / total if total else 0.0,
        }

    return results


if __name__ == "__main__":
    import sys
    import time

    from data_split import load_test_rows

    network_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Custom/Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID_custom.bin"
    )
    data_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "C:\AI_Verification\wireless\Pickle/mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D-003.pickle"
    )
    no_test_files = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    n_points = int(sys.argv[4]) if len(sys.argv) > 4 else None

    network = load_custom_network(network_path)
    X = load_test_rows(data_path, no_dataInFile=20000, no_test_files=no_test_files)
    if n_points is not None:
        X = X[:n_points]
    print(f"loaded {X.shape[0]} points (dim={X.shape[1]}) from {data_path} (test files={no_test_files})")

    # eps_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    eps_list = [4e-3, 6e-3, 8e-3]
    start_time = time.perf_counter()
    results = sweep(network, X, eps_list, k=8, max_diff=0)
    elapsed = time.perf_counter() - start_time

    for eps, r in results.items():
        print(f"eps={eps:>10}: T={r['T']:>4} F={r['F']:>4} ratio={r['ratio']:.3f}")

    print(f"total elapsed: {elapsed:.2f}s")
