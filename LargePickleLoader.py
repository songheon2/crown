"""
아주 큰(GB 단위) numpy ndarray가 담긴 pickle 파일에서, 전체를 메모리에 올리지 않고
필요한 행(row)만 부분적으로 읽어오는 유틸리티.

pickle 파일 전체를 pickle.load()로 읽으면 배열 전체가 메모리에 올라가야 하므로,
파일이 시스템 여유 메모리보다 훨씬 큰 경우(mMIMO 학습 데이터 등) 위험하다.

대신 pickle 스트림을 opcode 단위로 훑으면서(pickletools), ndarray의 실제 raw
데이터 바이트(가장 큰 부분)는 읽지 않고 건너뛰어(file.seek) 그 위치/길이만 알아낸다.
raw 데이터는 파일 안에서 하나의 연속된 C-contiguous 블록이므로, 이후
np.memmap으로 그 위치를 열어 필요한 행만 슬라이스해서 복사하면 된다.

현재는 "top-level 객체가 단일 numpy ndarray"인 pickle 파일만 지원한다
(mMIMO_AS_training_data_*.pickle 형식).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pickletools


@dataclass
class NdarrayPickleInfo:
    shape: Tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    data_offset: int
    data_length: int


class _LargeBlobEncountered(Exception):
    def __init__(self, pos: int, n: int):
        self.pos = pos
        self.n = n


class _SkipLargeReads:
    """file.read(n)이 threshold를 넘으면 실제로 읽지 않고 건너뛰는(seek) 래퍼."""

    def __init__(self, f, threshold: int):
        self.f = f
        self.threshold = threshold

    def read(self, n: int = -1):
        if isinstance(n, int) and n > self.threshold:
            pos = self.f.tell()
            self.f.seek(n, 1)
            raise _LargeBlobEncountered(pos, n)
        return self.f.read(n)

    def readline(self):
        return self.f.readline()

    def tell(self):
        return self.f.tell()


def peek_ndarray_info(filepath: str, threshold: int = 200_000) -> NdarrayPickleInfo:
    """
    pickle 파일을 opcode 단위로 훑어, 최상위 numpy ndarray의 shape/dtype/
    fortran_order와 raw 데이터의 (offset, length)를 raw 데이터를 읽지 않고 알아낸다.
    """
    shape: Tuple[int, ...] | None = None
    dtype: np.dtype | None = None
    fortran_order: bool | None = None
    data_offset: int | None = None
    data_length: int | None = None

    pending_ints: list[int] = []
    last_bool: bool | None = None
    expect_dtype_str = False

    with open(filepath, "rb") as f:
        while data_offset is None:
            wrapper = _SkipLargeReads(f, threshold)
            try:
                for opcode, arg, pos in pickletools.genops(wrapper):
                    name = opcode.name
                    if name in ("BININT", "BININT1", "BININT2"):
                        pending_ints.append(arg)
                    elif name in ("TUPLE", "TUPLE1", "TUPLE2", "TUPLE3"):
                        take = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}.get(name, len(pending_ints))
                        # take>=2 excludes numpy's dummy 1-element "(0,)" shape used
                        # internally to construct the empty placeholder ndarray
                        # before its real shape is applied via BUILD.
                        if shape is None and take >= 2 and len(pending_ints) >= take:
                            shape = tuple(pending_ints[-take:])
                        pending_ints.clear()
                    elif name == "STACK_GLOBAL":
                        expect_dtype_str = True
                    elif name == "SHORT_BINUNICODE":
                        if expect_dtype_str and dtype is None and arg not in ("dtype",):
                            try:
                                dtype = np.dtype(arg)
                            except TypeError:
                                pass
                        expect_dtype_str = False
                    elif name in ("NEWTRUE", "NEWFALSE"):
                        last_bool = name == "NEWTRUE"
                else:
                    break
            except _LargeBlobEncountered as e:
                data_offset = e.pos
                data_length = e.n
                fortran_order = bool(last_bool)
                break

    if shape is None or dtype is None or data_offset is None:
        raise ValueError(f"{filepath}: 최상위 ndarray 구조를 찾지 못했습니다.")

    expected = int(np.prod(shape)) * dtype.itemsize
    if expected != data_length:
        raise ValueError(
            f"파싱된 shape/dtype({shape}, {dtype})로 계산한 바이트 수({expected})가 "
            f"실제 skip한 데이터 길이({data_length})와 다릅니다."
        )

    return NdarrayPickleInfo(
        shape=shape,
        dtype=dtype,
        fortran_order=fortran_order,
        data_offset=data_offset,
        data_length=data_length,
    )


def load_rows(filepath: str, n_rows: int, info: NdarrayPickleInfo | None = None) -> np.ndarray:
    """앞에서부터 n_rows개 행만 복사해서 반환한다 (나머지는 메모리에 올리지 않음)."""
    if info is None:
        info = peek_ndarray_info(filepath)
    if info.fortran_order:
        raise NotImplementedError("fortran-order 배열의 부분 로딩은 지원하지 않습니다.")
    if len(info.shape) != 2:
        raise NotImplementedError("2차원 배열만 지원합니다.")

    mm = np.memmap(
        filepath,
        dtype=info.dtype,
        mode="r",
        offset=info.data_offset,
        shape=info.shape,
    )
    return np.array(mm[:n_rows])


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D-003.pickle"
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    info = peek_ndarray_info(filepath)
    print(info)

    rows = load_rows(filepath, n_rows, info=info)
    print("loaded shape:", rows.shape, "dtype:", rows.dtype)
    print("first row (first 8 values):", rows[0, :8])
