"""
CROWN(LiRPA) vs NeuralSAT의 mMIMO top-8 안테나 선택 로컬 강건성 결과를
idx x eps 단위로 교차 검증하는 스크립트.

배경 / 설계
-----------
NeuralSAT의 sat(반례 발견)은 VNNLIB 스펙 자체가 "exists i in clean_topk,
j not in clean_topk: Y_j >= Y_i"이므로, sat이라는 것 자체가 그 반례 지점에서
top-k 조합이 clean과 달라졌다는 뜻이다. 그래서 "sat인데 CROWN이 certified라고
주장하는 경우가 있는가"(예전 Task 1)와 "두 방법이 산출한 top-8이 idx x eps마다
같은가"(예전 Task 2)는 사실 같은 비교표에서 동시에 읽을 수 있는 두 관점이다.

idx x eps 조합마다 아래 두 값만 계산해서 비교한다:
  - crown_topk     : lb 기준 상위 k개 인덱스 (certified 여부와 무관하게 항상 계산됨)
  - neuralsat_topk : unsat이면 clean_topk(안 바뀜이 증명됨),
                     sat이면 result 파일의 counterexample x*를 CROWN 네트워크로
                     forward한 y*의 top-k

두 값이 다른 것 자체는 버그가 아니다(CROWN은 보수적 bound라 확신 못 할 뿐).
진짜 문제는 status=sat인데 crown_certified=True인 행("critical") — 실재하는
반례가 있는데 CROWN이 확실히 안전하다고 잘못 주장한 경우이며, 이 경우
crown_topk는 정의상 clean_topk와 같으므로 neuralsat_topk와 자동으로 불일치가
뜬다(별도 로직 없이 표 하나로 두 관점을 동시에 포착).

부가 검증: NeuralSAT은 onnx로, CROWN은 custom 바이너리로 같은 원본 모델을
읽으므로, 두 표현이 같은 함수인지도 매 행마다 clean 입력(x0)에서 대조한다
(model_clean_mismatch). sat 행은 result 파일에 같이 찍힌 Y_i(onnx 기준 y*)와
CROWN 네트워크로 x*를 다시 forward한 값도 대조한다(model_cex_mismatch).

하드코딩된 경로 (임시)
----------------------
NeuralSAT 실측 결과(summary.csv, manifest.csv, results/*.result)는 아직
jnunnv 저장소의 neuralsat/src/example/mmimo_topk에는 없고(파이프라인 코드만
있음), 별도 체크아웃 C:\\AI_Verification\\neuralsat 에만 실제로 쌓여 있다.
지금은 이 경로를 NEURALSAT_ROOT로 하드코딩해서 임시로 가져다 쓴다. 나중에
jnunnv 쪽에서 직접 sweep을 돌리게 되면 --summary/--manifest로 바꿔주면 된다.

사용법
------
    python compare_with_neuralsat.py [--sample-size 100] [--seed 0]
        [--k 8] [--method backward] [--out crown_vs_neuralsat.csv]
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort

from CustomToLirpa import load_custom_network
from pickle_memmap import load_row_range
from TopKRobustnessVerification import (
    _METHOD_VERIFIER_FACTORIES,
    certify_topk_selection,
    crown_bounds,
)

NEURALSAT_ROOT = Path(r"C:\AI_Verification\neuralsat\src\example\mmimo_topk")
DEFAULT_SUMMARY = NEURALSAT_ROOT / "summary.csv"
DEFAULT_MANIFEST = NEURALSAT_ROOT / "vnnlib" / "manifest.csv"

DEFAULT_CROWN_NETWORK = (
    Path(__file__).resolve().parent.parent
    / "models" / "Custom"
    / "Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID_custom.bin"
)

_CEX_VAR_RE = re.compile(r"\(([XY])_(\d+)\s+([^)]+)\)")


def parse_cex(result_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """result 파일에서 counterexample x*와 그때 NeuralSAT이 찍은 y*(onnx 기준)를 파싱한다."""
    text = result_path.read_text()
    xs: dict[int, float] = {}
    ys: dict[int, float] = {}
    for kind, idx_s, val_s in _CEX_VAR_RE.findall(text):
        (xs if kind == "X" else ys)[int(idx_s)] = float(val_s)
    if not xs:
        raise ValueError(f"'{result_path}'에서 counterexample을 찾지 못했습니다")
    x_star = np.array([xs[i] for i in range(len(xs))], dtype=np.float64)
    y_star = np.array([ys[i] for i in range(len(ys))], dtype=np.float64) if ys else None
    return x_star, y_star


def topk_set(y: np.ndarray, k: int) -> frozenset:
    return frozenset(np.argsort(y)[-k:].tolist())


def lb_topk_set(lb: np.ndarray, k: int) -> frozenset:
    return frozenset(np.argsort(lb)[::-1][:k].tolist())


def sample_summary_rows(summary_path: Path, sample_size: int, seed: int) -> list[dict]:
    """summary.csv(수십만 행)를 한 번만 스트리밍하며 status in {sat, unsat}인 행에서
    reservoir sampling으로 sample_size개를 뽑는다 (전체를 메모리에 올리지 않음)."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] not in ("sat", "unsat"):
                continue
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_size:
                    reservoir[j] = row
    return reservoir


def load_manifest_rows(manifest_path: Path, keys: set) -> dict:
    """manifest.csv를 스트리밍하며 필요한 (idx, eps) 키만 골라 담는다."""
    found: dict = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["idx"]), float(row["eps"]))
            if key in keys:
                found[key] = row
    return found


def classify(status: str, crown_certified: bool, topk_match: bool) -> str:
    if status == "sat" and crown_certified:
        return "critical"  # 실재하는 반례가 있는데 CROWN이 certified라고 주장 -> 버그 후보
    if status == "sat":
        return "match" if topk_match else "expected_mismatch"  # CROWN이 확신 못 한 것뿐, 정상
    # status == "unsat"
    if topk_match:
        return "match"
    if crown_certified:
        return "critical"  # certified면 정의상 같아야 하는데 다름 -> 버그 후보
    return "sanity_gap"  # CROWN bound가 헐거운 것뿐, 정상


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--crown-network", default=str(DEFAULT_CROWN_NETWORK))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--method", default="backward", choices=sorted(_METHOD_VERIFIER_FACTORIES))
    parser.add_argument("--out", default="crown_vs_neuralsat.csv")
    args = parser.parse_args()

    print(f"summary.csv 스캔 및 샘플링 중: {args.summary}")
    sampled = sample_summary_rows(Path(args.summary), args.sample_size, args.seed)
    print(f"  {len(sampled)}개 (idx, eps) 샘플 확보")
    if not sampled:
        print("샘플이 없습니다 (sat/unsat 행이 없음). 종료.")
        return

    keys = {(int(r["idx"]), float(r["eps"])) for r in sampled}
    print(f"manifest.csv에서 매칭 행 조회 중: {args.manifest}")
    manifest_rows = load_manifest_rows(Path(args.manifest), keys)

    print(f"CROWN 네트워크 로딩: {args.crown_network}")
    crown_net = load_custom_network(args.crown_network)

    ort_sessions: dict[str, ort.InferenceSession] = {}

    def get_session(net_path: str) -> ort.InferenceSession:
        sess = ort_sessions.get(net_path)
        if sess is None:
            sess = ort.InferenceSession(net_path, providers=["CPUExecutionProvider"])
            ort_sessions[net_path] = sess
        return sess

    rows_out = []
    critical_rows = []

    for i, srow in enumerate(sampled):
        idx = int(srow["idx"])
        eps = float(srow["eps"])
        status = srow["status"]
        key = (idx, eps)
        mrow = manifest_rows.get(key)
        if mrow is None:
            print(f"경고: manifest에 (idx={idx}, eps={eps}) 없음, 건너뜀")
            continue

        abs_row = int(mrow["abs_row"])
        data_path = mrow["data"]
        net_path = mrow["net"]

        x0 = load_row_range(data_path, abs_row, abs_row + 1)[0].astype(np.float64)

        y0_custom = crown_net.forward(x0)
        clean_topk_custom = topk_set(y0_custom, args.k)

        sess = get_session(net_path)
        input_name = sess.get_inputs()[0].name
        y0_onnx = sess.run(None, {input_name: x0.astype(np.float32).reshape(1, -1)})[0].reshape(-1)
        clean_topk_onnx = topk_set(y0_onnx, args.k)
        model_clean_mismatch = clean_topk_custom != clean_topk_onnx

        lb, ub = crown_bounds(crown_net, x0, eps, method=args.method)
        crown_topk = lb_topk_set(lb, args.k)
        cert = certify_topk_selection(y0_custom, lb, ub, k=args.k, max_diff=0)

        cex_outside_ball = None
        model_cex_mismatch = None
        if status == "sat":
            result_path = Path(mrow["result"])
            try:
                x_star, y_star_reported = parse_cex(result_path)
            except ValueError:
                # run_batch.py의 eps 조기 종료("implied sat")로, 이 정확한 eps에서
                # 독립적으로 검증된 counterexample이 없는 경우. 스킵.
                print(f"  건너뜀: idx={idx} eps={eps}는 implied sat(cex 없음)")
                continue
            max_dev = float(np.abs(x_star - x0).max())
            cex_outside_ball = max_dev > eps + 1e-6
            y_star_custom = crown_net.forward(x_star)
            neuralsat_topk = topk_set(y_star_custom, args.k)
            if y_star_reported is not None:
                model_cex_mismatch = topk_set(y_star_reported, args.k) != neuralsat_topk
        else:
            neuralsat_topk = clean_topk_custom

        topk_match = crown_topk == neuralsat_topk
        severity = classify(status, bool(cert["certified"]), topk_match)

        row_out = {
            "idx": idx,
            "eps": eps,
            "neuralsat_status": status,
            "neuralsat_runtime": srow["runtime"],
            "crown_certified": cert["certified"],
            "crown_gap_ok": cert["gap_ok"],
            "topk_match": topk_match,
            "severity": severity,
            "model_clean_mismatch": model_clean_mismatch,
            "model_cex_mismatch": model_cex_mismatch,
            "cex_outside_eps_ball": cex_outside_ball,
            "crown_topk": sorted(crown_topk),
            "neuralsat_topk": sorted(neuralsat_topk),
        }
        rows_out.append(row_out)
        if severity == "critical":
            critical_rows.append(row_out)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sampled)} 처리 완료")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"결과 저장: {args.out} ({len(rows_out)}행)")

    counts = Counter(r["severity"] for r in rows_out)
    print(f"심각도 분포: {dict(counts)}")
    model_clean_mismatches = sum(1 for r in rows_out if r["model_clean_mismatch"])
    model_cex_mismatches = sum(1 for r in rows_out if r["model_cex_mismatch"])
    if model_clean_mismatches:
        print(f"경고: onnx/custom-bin clean top-k 불일치 {model_clean_mismatches}건 (모델 변환 자체를 의심할 것)")
    if model_cex_mismatches:
        print(f"경고: onnx/custom-bin counterexample top-k 불일치 {model_cex_mismatches}건")

    if critical_rows:
        print(f"\n=== critical 상세 진단 ({len(critical_rows)}건) ===")
        for row_out in critical_rows:
            idx, eps = row_out["idx"], row_out["eps"]
            mrow = manifest_rows[(idx, eps)]
            x0 = load_row_range(mrow["data"], int(mrow["abs_row"]), int(mrow["abs_row"]) + 1)[0].astype(np.float64)
            x_star, _ = parse_cex(Path(mrow["result"]))
            lb, ub = crown_bounds(crown_net, x0, eps, method=args.method)
            y_star = crown_net.forward(x_star)
            print(f"idx={idx} eps={eps}")
            print(f"  max|x*-x0| = {float(np.abs(x_star - x0).max()):.3e} (eps={eps}, ball 밖={row_out['cex_outside_eps_ball']})")
            for oi in range(len(y_star)):
                violated = not (lb[oi] - 1e-9 <= y_star[oi] <= ub[oi] + 1e-9)
                mark = "  <-- bound 위반!" if violated else ""
                print(f"  Y_{oi}: lb={lb[oi]:.6f} ub={ub[oi]:.6f} y*={y_star[oi]:.6f}{mark}")
    else:
        print("\ncritical 케이스 없음")


if __name__ == "__main__":
    main()
