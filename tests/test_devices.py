import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from turnalign.cli import main
from turnalign.devices import Device, backend_plan, detect_devices, select_device
from turnalign.plugins import Accelerator


class FakeCuda:
    def __init__(self, available=False, names=()):
        self.available = available
        self.names = tuple(names)

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.names)

    def get_device_name(self, index):
        return self.names[index]

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=(index + 1) * 8 * 1024**3)


class FakeMps:
    def __init__(self, available=False):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    __version__ = "test"

    def __init__(self, *, hip=None, cuda=None, mps=False):
        self.version = SimpleNamespace(hip=hip, cuda=cuda)
        self.cuda = FakeCuda(bool(cuda or hip), ("GPU 0", "GPU 1") if cuda or hip else ())
        self.backends = SimpleNamespace(mps=FakeMps(mps))


class FakeOrt:
    def __init__(self, providers):
        self.providers = providers

    def get_available_providers(self):
        return self.providers


class DeviceDetectionTests(unittest.TestCase):
    def test_nvidia_cuda_is_detected(self):
        devices = detect_devices(
            FakeTorch(cuda="13.0"),
            FakeOrt(("CUDAExecutionProvider", "CPUExecutionProvider")),
        )
        gpu = devices[0]
        self.assertEqual(gpu.accelerator, Accelerator.CUDA)
        self.assertEqual((gpu.vendor, gpu.device, gpu.dtype), ("NVIDIA", "cuda:0", "float16"))
        self.assertEqual(gpu.memory_bytes, 8 * 1024**3)
        self.assertEqual(gpu.onnx_providers, ("CUDAExecutionProvider",))

    def test_amd_rocm_uses_pytorch_cuda_device_syntax(self):
        devices = detect_devices(
            FakeTorch(hip="7.2"),
            FakeOrt(("ROCMExecutionProvider", "CPUExecutionProvider")),
        )
        gpu = devices[0]
        self.assertEqual(gpu.accelerator, Accelerator.ROCM)
        self.assertEqual((gpu.vendor, gpu.device), ("AMD", "cuda:0"))
        self.assertIn("ROCm 7.2", gpu.runtime)

    def test_apple_mps_is_detected(self):
        devices = detect_devices(
            FakeTorch(mps=True),
            FakeOrt(("CoreMLExecutionProvider", "CPUExecutionProvider")),
        )
        gpu = devices[0]
        self.assertEqual(gpu.accelerator, Accelerator.MPS)
        self.assertEqual((gpu.vendor, gpu.device), ("Apple", "mps"))
        self.assertEqual(gpu.onnx_providers, ("CoreMLExecutionProvider",))

    def test_cpu_is_always_present(self):
        devices = detect_devices(FakeTorch(), FakeOrt(("CPUExecutionProvider",)))
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].accelerator, Accelerator.CPU)


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.devices = [
            Device(Accelerator.ROCM, True, "ROCm", "cuda:0", "AMD", "Radeon", "float16"),
            Device(Accelerator.CPU, True, "native", "cpu", "x86", "CPU", "float32"),
        ]

    def test_auto_prefers_gpu(self):
        self.assertEqual(select_device(devices=self.devices).accelerator, Accelerator.ROCM)

    def test_auto_prefers_cuda_when_both_gpu_runtimes_exist(self):
        cuda = Device(Accelerator.CUDA, True, "CUDA", "cuda:0", "NVIDIA", "RTX", "float16")
        self.assertEqual(select_device(devices=self.devices + [cuda]).accelerator, Accelerator.CUDA)

    def test_backend_capability_can_force_cpu(self):
        selected = select_device(devices=self.devices, supported=(Accelerator.CPU,))
        self.assertEqual(selected.accelerator, Accelerator.CPU)

    def test_explicit_unavailable_device_fails(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            select_device("cuda", devices=self.devices)

    def test_device_index_is_respected(self):
        second = Device(Accelerator.ROCM, True, "ROCm", "cuda:1", "AMD", "Radeon 1", "float16")
        self.assertEqual(select_device("rocm:1", devices=self.devices + [second]).device, "cuda:1")

    def test_environment_override_wins(self):
        with patch.dict(os.environ, {"TURNALIGN_DEVICE": "cpu"}):
            self.assertEqual(select_device("rocm", devices=self.devices).accelerator, Accelerator.CPU)

    def test_rocm_plan_preserves_cuda_device_string(self):
        plan = backend_plan(self.devices[0])
        self.assertEqual(plan["pytorch"]["device"], "cuda:0")
        self.assertIn("faster-whisper/CTranslate2 GPU", plan["unsupported_engines"])

    def test_mps_plan_enables_operator_fallback(self):
        mps = Device(Accelerator.MPS, True, "MPS", "mps", "Apple", "M3", "float16")
        self.assertEqual(backend_plan(mps)["environment"]["PYTORCH_ENABLE_MPS_FALLBACK"], "1")


class DoctorCliTests(unittest.TestCase):
    def test_doctor_prints_machine_readable_report(self):
        output = StringIO()
        with patch("sys.argv", ["turnalign", "doctor", "--device", "cpu"]), redirect_stdout(output):
            self.assertEqual(main(), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["selected"]["accelerator"], "cpu")
        self.assertIn("backend_plan", report)
        self.assertIn("platform", report)


if __name__ == "__main__":
    unittest.main()
