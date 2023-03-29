from compute import block_
from compute.utils import AEQB
import kokkos
import numpy as np


class b(block_):
    def __init__(self):
        # The c++ stuff must be instantiated first,
        # so that inhereted python side
        # attributes are assigned values, not defined
        # in the upstream __init__s
        block_.__init__(self)
        self.x = kokkos.array(
            "x",
            shape=(10, 10, 10),
            layout=kokkos.LayoutLeft,
            dtype=kokkos.double,
            space=kokkos.CudaSpace,
            dynamic=False,
        )

        self.y = kokkos.array(
            "y",
            shape=(10, 10, 10),
            layout=kokkos.LayoutLeft,
            dtype=kokkos.double,
            space=kokkos.CudaSpace,
            dynamic=False,
        )

        self.x_m = kokkos.create_mirror_view(self.x, copy=True)
        self.y_m = kokkos.create_mirror_view(self.y, copy=True)

        self.x_np = np.array(self.x_m)
        self.y_np = np.array(self.y_m)


def simulate():
    B = b()

    B.x_np[:] = np.zeros((10, 10, 10))

    kokkos.deep_copy(B.x, B.x_m)
    print("Copy Completed!")

    AEQB(B.x, B.y)
    print("Compute Kernel Completed!")


if __name__ == "__main__":
    try:
        kokkos.initialize()
        simulate()
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
