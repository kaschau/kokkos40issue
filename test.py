from compute import block_, view3, initialize, finalize
from compute.utils import AEQB
import numpy as np


class b(block_):
    def __init__(self):
        # The c++ stuff must be instantiated first,
        # so that inhereted python side
        # attributes are assigned values, not defined
        # in the upstream __init__s
        block_.__init__(self)
        self.x = view3(
            "x",
            *(10, 10, 10),
        )

        self.y = view3(
            "y",
            *(10, 10, 10),
        )


def simulate():
    B = b()

    AEQB(B.x, B.y)
    print("Compute Kernel Completed!")


if __name__ == "__main__":
    try:
        initialize()
        simulate()
        finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
