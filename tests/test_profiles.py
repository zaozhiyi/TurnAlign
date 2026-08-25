import unittest

from turnalign.components.funasr import _pytorch_device
from turnalign.devices import Device
from turnalign.plugins import Accelerator
from turnalign.profiles import PROFILE_NAMES, profile_catalog, select_execution_profile


def device(accelerator, target, vendor="test"):
    return Device(accelerator, True, accelerator.value, target, vendor, target, "float16")


CPU = device(Accelerator.CPU, "cpu")


class ExecutionProfileTests(unittest.TestCase):
    def test_auto_selects_mac_balanced(self):
        profile = select_execution_profile(
            "auto", devices=[device(Accelerator.MPS, "mps", "Apple"), CPU], system="Darwin"
        )
        self.assertEqual(profile.name, "mac-balanced")
        self.assertTrue(profile.parallel_diarization)
        self.assertEqual(profile.alignment_batch_size, 4)

    def test_auto_selects_single_and_multi_cuda_profiles(self):
        one = [device(Accelerator.CUDA, "cuda:0", "NVIDIA"), CPU]
        two = [device(Accelerator.CUDA, "cuda:0", "NVIDIA"), device(
            Accelerator.CUDA, "cuda:1", "NVIDIA"
        ), CPU]
        self.assertEqual(
            select_execution_profile("auto", devices=one, system="Linux").name,
            "cuda-single-gpu",
        )
        multi = select_execution_profile("auto", devices=two, system="Linux")
        self.assertEqual(multi.name, "cuda-multi-gpu")
        self.assertEqual((multi.asr_device, multi.diarization_device), ("cuda:0", "cuda:1"))
        self.assertTrue(multi.parallel_diarization)

    def test_auto_selects_platform_specific_rocm_profile(self):
        devices = [device(Accelerator.ROCM, "cuda:0", "AMD"), CPU]
        linux = select_execution_profile("auto", devices=devices, system="Linux")
        windows = select_execution_profile("auto", devices=devices, system="Windows")
        self.assertEqual(linux.name, "rocm-linux")
        self.assertEqual(windows.name, "rocm-windows")
        self.assertEqual(linux.alignment_device, "cuda:0")
        self.assertEqual(windows.alignment_device, "cpu")
        self.assertEqual(windows.alignment_batch_size, 1)

    def test_explicit_device_guides_auto_profile(self):
        devices = [device(Accelerator.CUDA, "cuda:0", "NVIDIA"), CPU]
        profile = select_execution_profile(
            "auto", requested_device="cpu", devices=devices, system="Linux"
        )
        self.assertEqual(profile.name, "cpu-low-memory")

    def test_explicit_gpu_index_uses_single_device_profile(self):
        devices = [
            device(Accelerator.CUDA, "cuda:0", "NVIDIA"),
            device(Accelerator.CUDA, "cuda:1", "NVIDIA"),
            CPU,
        ]
        profile = select_execution_profile(
            "auto", requested_device="cuda:1", devices=devices, system="Linux"
        )
        self.assertEqual(profile.name, "cuda-single-gpu")
        self.assertEqual(
            {profile.asr_device, profile.vad_device, profile.alignment_device,
             profile.diarization_device},
            {"cuda:1"},
        )

    def test_unavailable_or_wrong_platform_profile_fails(self):
        with self.assertRaisesRegex(RuntimeError, "requires 2 cuda"):
            select_execution_profile(
                "cuda-multi-gpu",
                devices=[device(Accelerator.CUDA, "cuda:0"), CPU],
                system="Linux",
            )
        with self.assertRaisesRegex(RuntimeError, "requires Windows"):
            select_execution_profile(
                "rocm-windows",
                devices=[device(Accelerator.ROCM, "cuda:0"), CPU],
                system="Linux",
            )

    def test_catalog_is_serializable_and_complete(self):
        catalog = profile_catalog()
        self.assertEqual(len(catalog), len(PROFILE_NAMES) - 1)
        self.assertTrue(all(isinstance(item["notes"], list) for item in catalog))

    def test_rocm_component_device_uses_pytorch_cuda_syntax(self):
        self.assertEqual(_pytorch_device("rocm:1"), "cuda:1")
        self.assertEqual(_pytorch_device("cuda:1"), "cuda:1")


if __name__ == "__main__":
    unittest.main()
