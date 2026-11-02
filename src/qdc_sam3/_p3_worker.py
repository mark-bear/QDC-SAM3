"""Private subprocess entry point for bounded OpenCV Selective Search."""

from __future__ import annotations

import pickle
import sys
import traceback

import cv2

from .proposals import _p3


def main() -> None:
    try:
        cv2.setNumThreads(1)
        cv2.setRNGSeed(20260903)
        ir_u8, config = pickle.loads(sys.stdin.buffer.read())
        response = True, _p3(ir_u8, config)
    except Exception:  # noqa: BLE001 - the parent process must receive worker failures
        response = False, traceback.format_exc()
    sys.stdout.buffer.write(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()
