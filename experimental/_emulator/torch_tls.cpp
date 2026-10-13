#include <ATen/ThreadLocalState.h>
#include <torch/extension.h>


class GreenletThreadLocalState {
 public:
  GreenletThreadLocalState() = default;

  void restore() const {
    at::ThreadLocalState::setThreadLocalState(state_);
  }

 private:
  at::ThreadLocalState state_;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<GreenletThreadLocalState>(module, "ThreadLocalState")
      .def(pybind11::init<>())
      .def("restore", &GreenletThreadLocalState::restore);
}
