// Reference: https://github.com/pytorch/pytorch/blob/master/torch/csrc/cuda/nccl.cpp

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <nccl.h>
#include <torch/extension.h>

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace torch;
using namespace std;

ncclComm_t *to_nccl_comm(ncclComm_t *var) { return reinterpret_cast<ncclComm_t *>(var); }

ncclComm_t to_nccl_comm(ncclComm_t var) { return reinterpret_cast<ncclComm_t>(var); }

ncclDataType_t to_nccl_data_type(c10::ScalarType type) {
    switch (type) {
    case at::kFloat:
        return ncclDataType_t::ncclFloat;
    case at::kHalf:
        return ncclDataType_t::ncclHalf;
    case at::kDouble:
        return ncclDataType_t::ncclDouble;
    case at::kLong:
        return ncclDataType_t::ncclInt64;
    case at::kInt:
        return ncclDataType_t::ncclInt;
    case at::kChar:
        return ncclDataType_t::ncclChar;
    case at::kByte:
        return ncclDataType_t::ncclUint8;
    case at::kBool:
        return ncclDataType_t::ncclUint8;
#if HAS_NCCL_BF16_DATATYPE
    case at::kBFloat16:
        return ncclDataType_t::ncclBfloat16;
#endif
    default:
        TORCH_CHECK(false, "Unconvertible NCCL type ", type);
    }
}

ncclDataType_t to_nccl_data_type(const at::Tensor &t) {
    if (!t.is_cuda()) {
        TORCH_CHECK(false, "NCCL only supports CUDA tensors, but got a tensor on ", t.device());
    }
    return to_nccl_data_type(t.scalar_type());
}

ncclRedOp_t to_nccl_red_op(int var) { return (ncclRedOp_t)(var); }

void throw_nccl_error(ncclResult_t status) {
    std::ostringstream err;
    err << "NCCL Error " << static_cast<int>(status) << ": " << ncclGetErrorString(status);
    throw std::runtime_error(err.str());
}

static inline void NCCL_CHECK(ncclResult_t status) {
    if (status != ncclSuccess) {
        throw_nccl_error(status);
    }
}

void comm_destroy(ncclComm_t comm) {
    // DO NOT DO ANYTHING.
    return;
}

struct AutoNcclGroup {
    AutoNcclGroup() {
#if (TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR < 14)
        (c10::cuda::CUDACachingAllocator::getFreeMutex())->lock();
#else
        (c10::cuda::getFreeMutex())->lock();
#endif
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
        NCCL_CHECK(ncclGroupStart());
#endif
    }
    ~AutoNcclGroup() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
        NCCL_CHECK(ncclGroupEnd());
#endif
#if (TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR < 14)
        (c10::cuda::CUDACachingAllocator::getFreeMutex())->unlock();
#else
        (c10::cuda::getFreeMutex())->unlock();
#endif
    }
};

struct NcclCommList {
    std::unique_ptr<ncclComm_t[]> comms;
    int ndevices;
    NcclCommList(const std::vector<int> &devices) : comms(new ncclComm_t[devices.size()]), ndevices(devices.size()) {
        NCCL_CHECK(ncclCommInitAll(to_nccl_comm(comms.get()), devices.size(), devices.data()));
    }
    NcclCommList(NcclCommList &&foo) = default;
    ~NcclCommList() {
        if (comms) {
            for (const auto i : c10::irange(ndevices)) {
                int dummy_var;
                if (cudaGetDevice(&dummy_var) != cudaSuccess) {
                    /* there are cases when this destructor is called after the
                     CUDA driver is already unloaded from the process.
                     In these cases, skip ncclCommDestroy. */
                    return;
                }
                comm_destroy(comms[i]);
            }
        }
    }
    ArrayRef<ncclComm_t> ref() const { return ArrayRef<ncclComm_t>(comms.get(), ndevices); }
};

using device_list = std::vector<int>;
// accesses to this object have to be guarded by THC's CudaFreeMutex.
static std::unordered_map<device_list, NcclCommList, c10::hash<device_list>> _communicators;

struct ncclCommPtr {
    ncclComm_t ptr;
};

ArrayRef<ncclComm_t> get_communicators(TensorList inputs) {
    static auto get_device = [](const at::Tensor &t) -> int { return t.get_device(); };
    device_list devices = fmap(inputs, get_device);
    auto it = _communicators.find(devices);
    if (it == _communicators.end())
        std::tie(it, std::ignore) = _communicators.emplace(devices, devices);
    return it->second.ref();
}

ncclUniqueId get_nccl_uid() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    return id;
}

ncclCommPtr get_communicator(ncclUniqueId uid, int rank, int world_size) {
    ncclCommPtr comm;
    ncclCommInitRank(&comm.ptr, world_size, uid, rank);
    return comm;
}

void dist_reduce(const at::Tensor &input, at::Tensor &output, int32_t root, int32_t op, const ncclCommPtr &user_comm,
            int nccl_type = -1) {
    ncclDataType_t data_type = nccl_type == -1 ? to_nccl_data_type(input) : (ncclDataType_t)nccl_type;

    const auto count = input.numel();

    AutoNcclGroup nccl_group_guard;
    at::cuda::OptionalCUDAGuard device_guard;
    int device = input.device().index();
    device_guard.set_index(device);
    // Default to the current stream.
    const auto stream = at::cuda::getCurrentCUDAStream(device).stream();
    ncclComm_t comm = user_comm.ptr;
    NCCL_CHECK(ncclReduce(input.data_ptr(), output.data_ptr(), count, data_type, to_nccl_red_op(op), root,
                          to_nccl_comm(comm), stream));
}

void dist_all_reduce(const vector<Tensor> &inputs, vector<Tensor> &outputs, int32_t op,
                const std::vector<ncclCommPtr> &user_comms, int nccl_type = -1) {
    const auto len = inputs.size();

    ncclDataType_t data_type = nccl_type == -1 ? to_nccl_data_type(inputs[0]) : (ncclDataType_t)nccl_type;

    const auto count = inputs[0].numel();

    AutoNcclGroup nccl_group_guard;
    at::cuda::OptionalCUDAGuard device_guard;
    for (const auto i : c10::irange(len)) {
        int device = inputs[i].device().index();
        device_guard.set_index(device);
        // Default to the current stream.
        const auto stream = at::cuda::getCurrentCUDAStream(device).stream();

        ncclComm_t comm = user_comms[i].ptr;
        NCCL_CHECK(ncclAllReduce(inputs[i].data_ptr(), outputs[i].data_ptr(), count, data_type, to_nccl_red_op(op),
                                 to_nccl_comm(comm), stream));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ncclUniqueId>(m, "ncclUniqueId")
        .def(py::init<>())
        .def(py::pickle(
            [](const ncclUniqueId &id) { return py::bytes(reinterpret_cast<const char *>(&id), sizeof(id)); },
            [](py::bytes b) {
                const std::string s = b;
                if (s.size() != sizeof(ncclUniqueId)) {
                    throw std::runtime_error("Invalid state!");
                }
                ncclUniqueId id;
                memcpy(&id, s.data(), sizeof(id));
                return id;
            }));
    py::class_<ncclCommPtr>(m, "ncclCommPtr").def(py::init<>());
    m.def("reduce", &dist_reduce, "Reduce");
    m.def("all_reduce", &dist_all_reduce, "All reduce");
    m.def("get_nccl_uid", &get_nccl_uid, "Get NCCL UID");
    m.def("get_communicator", &get_communicator, "Get communicator");
}
