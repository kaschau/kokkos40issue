#ifndef __kokkosTypes_H__
#define __kokkosTypes_H__

#include "Kokkos_Core.hpp"

// Define the execution and storage space
#if defined(KOKKOS_ENABLE_CUDA)
using execSpace = Kokkos::Cuda;
using viewSpace = Kokkos::CudaSpace;
using layout = Kokkos::LayoutLeft;
static const std::string KokkosLocation = "Cuda";
#elif defined(KOKKOS_ENABLE_SERIAL)
using execSpace = Kokkos::Serial;
using viewSpace = Kokkos::HostSpace;
using layout = Kokkos::LayoutRight;
static const std::string KokkosLocation = "Serial";
#endif

using defaultViewHooks = Kokkos::Experimental::DefaultViewHooks;
// define some shorthand for the Kokkos views and Range Policies
using threeDview =
    Kokkos::View<double ***, layout, viewSpace, defaultViewHooks>;
using MDRange3 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<3>>;
#endif
