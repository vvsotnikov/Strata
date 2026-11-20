"""Hy4 follows the ordinary catalog/import/install path without a Qwen engine."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from strata_control import acquisition, hy4, hy4_types
from tests.terminal.launch_support import HW_24, Home, catalog

TOOLS = Path(__file__).resolve().parents[2] / "tools"
from hy4 import prepare as prepare_module

CAPS = {"architecture": hy4.ARCH, "backend": "cuda", "features":
        {"batching": True, "vision": True, "mtp": True, "kv_gpu": False}}


class NativeHy4Catalog(unittest.TestCase):
    def test_own_path_resolves_size_and_full_context_range(self):
        with Home():
            cat = catalog(HW_24)
            self.assertEqual(cat.resolve("q4_k_m", "hy4"), ("hy4", "Q4_K_M"))
            self.assertEqual(cat.contexts_for("hy4")[-1], 1048576)
            row = next(f for f in cat.strata.model_table(cat.hardware()) if f["family"] == "hy4")
            self.assertEqual(len(row["sizes"]), 1)
            size = row["sizes"][0]
            self.assertTrue(size["source_only"])
            self.assertTrue(size["multi_file"])
            self.assertEqual(size["images_reader"], "cpu")
            self.assertIsNone(size["engine_min"])
            self.assertEqual(cat.strata.own_size("hy4", "Q4_K_M")["engine"], None)

    def test_expected_files_include_engram_shard_and_projector(self):
        import strata_mcp
        with Home():
            cat = catalog(HW_24)
            rows = strata_mcp.expected_files(cat.strata.setup_module(), "hy4", "Q4_K_M", Path("/models"), True)
            self.assertEqual([Path(r["path"]).name for r in rows], [*hy4.FILES, hy4.PROJECTOR])
            self.assertEqual([r["bytes"] for r in rows], [hy4.PINS[n][0] for n in [*hy4.FILES, hy4.PROJECTOR]])

    def test_source_build_is_separate_and_arch_specific(self):
        commands = hy4.build_commands(TOOLS.parent, "cuda", ["86", "sm_86"])
        self.assertIn("-DSTRATA_ENABLE_CUDA=OFF", commands[0])
        self.assertIn("-DSTRATA_HY4_CUDA=ON", commands[0])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", commands[0])
        self.assertEqual(commands[1][-1], "-j4")
        with self.assertRaisesRegex(ValueError, "known compute"):
            hy4.build_commands(TOOLS.parent, "cuda", [])
        with self.assertRaisesRegex(ValueError, "native CPU"):
            hy4.build_commands(TOOLS.parent, "hip", [])

    def test_capabilities_fail_closed_without_loading_weights(self):
        done = SimpleNamespace(returncode=0, stdout=json.dumps(CAPS))
        with patch("strata_control.hy4.subprocess.run", return_value=done) as run:
            self.assertEqual(hy4.capabilities(Path("/bin/native")), CAPS)
        self.assertEqual(run.call_args.args[0], ["/bin/native", "--capabilities"])
        done.stdout = json.dumps(CAPS | {"features": {"vision": "false"}})
        with patch("strata_control.hy4.subprocess.run", return_value=done), self.assertRaisesRegex(ValueError, "malformed"):
            hy4.capabilities(Path("/bin/native"))

    def test_physical_cards_resolve_to_uuids_without_cuda_order_assumptions(self):
        done = SimpleNamespace(returncode=0, stdout="3, GPU-third\n2, GPU-second\n0, GPU-other\n")
        with patch("strata_control.hy4.subprocess.run", return_value=done):
            self.assertEqual(hy4.cuda_visible_devices([2, 3]), "GPU-second,GPU-third")
            with self.assertRaisesRegex(ValueError, "not available"):
                hy4.cuda_visible_devices([4])

    def test_launch_plan_names_source_build_and_embedded_drafter(self):
        from strata_cli.launch import settings
        from strata_cli.launch.runner import Runner, derive_config
        with Home():
            cat = catalog(HW_24)
            s = settings.LaunchSettings(family="hy4", model="Q4_K_M", context=1048576,
                                       parallel=4, vision="cpu", drafting="mtp", gpus=[2, 3])
            runner = Runner(s, catalog=cat)
            with patch.object(runner, "intel_chosen", return_value=False):
                plan = runner.plan()
            self.assertEqual(plan["warnings"], [])
            self.assertIn("native MTP", plan["drafting"])
            self.assertNotIn("DFlash2", plan["download"]["text"])
            self.assertIn("no published engine component", plan["download"]["text"])
            self.assertEqual({p["kind"] for p in plan["download"]["parts"]}, {"weights", "mmproj"})
            base = {"arch": hy4.ARCH, "backend": "cuda", "args": ["--cuda-devices", "0", "--slots", "1", "--mtp", "0"], "mtp": False}
            cfg = derive_config(base, s, bool_flags=frozenset(), gpus=[2, 3])
            self.assertEqual(cfg["parallel"], 4)
            self.assertNotIn("layer_split", cfg)
            self.assertEqual(cfg["args"][cfg["args"].index("--cuda-devices") + 1], "0,1")
            self.assertEqual(cfg["args"][cfg["args"].index("--mtp") + 1], "2")

    def test_native_advanced_settings_and_preview_do_not_become_qwen_flags(self):
        from serve.server import apply_vision_defaults, engine_args
        from strata_cli.launch import settings
        from strata_cli.launch.runner import apply_choices, derive_config
        base = {"arch": hy4.ARCH, "backend": "cuda", "args": ["--max-context", "8192", "--context-pool", "8192", "--slots", "1", "--mtp", "0", "--mmproj", "/projector"], "mtp": False,
                "vision": {"mmproj": "/projector", "gpu": True}}
        s = settings.LaunchSettings(family="hy4", model="Q4_K_M", parallel=4, extra={"--context-pool": "1m", "--mtp": "5"})
        cfg = derive_config(base, s, bool_flags=frozenset())
        self.assertEqual(cfg["parallel"], 4)
        self.assertEqual(cfg["args"][cfg["args"].index("--slots")+1], "4")
        self.assertEqual(cfg["context_pool"], 1048576)
        self.assertEqual(cfg["drafting"], "mtp")
        self.assertTrue(cfg["mtp"])
        actual = engine_args(cfg)
        self.assertEqual(actual[actual.index("--context-pool")+1], "1048576")
        self.assertEqual(actual[actual.index("--mtp")+1], "5")
        on = derive_config(base, settings.LaunchSettings(family="hy4", drafting="auto"), bool_flags=frozenset())
        self.assertEqual(on["args"][on["args"].index("--mtp")+1], "2")
        preview = apply_choices(base, settings.LaunchSettings(family="hy4", vision="cpu", kv="fp16"))
        self.assertNotIn("--vision", preview["args"])
        self.assertNotIn("--kv", preview["args"])
        self.assertIn("--mmproj", preview["args"])
        self.assertNotIn("elastic", apply_vision_defaults(base)["vision"])

    def test_trunk_kquant_and_bf16_mtp_are_distinct(self):
        types = {n: 0 for n in ("token_embd.weight", "output.weight", "output_norm.weight", "output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight")}
        types.update({"per_layer_token_embd.weight": 43, "blk.0.ffn_gate_exps.weight": 12,
                      "blk.0.ffn_up_exps.weight": 12, "blk.0.ffn_down_exps.weight": 14,
                      "blk.42.ffn_gate_exps.weight": 30, "blk.42.ffn_up_exps.weight": 30, "blk.42.ffn_down_exps.weight": 30})
        self.assertEqual(hy4_types.problems(types), [])
        self.assertTrue(hy4_types.problems(types | {"blk.0.ffn_gate_exps.weight": 30}))
        self.assertTrue(hy4_types.problems(types | {"per_layer_token_embd.weight": 0}))


class NativeHy4Preparation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = self.root / "strata-hy4"
        self.engine.touch()
        self.model = self.root / hy4.FILES[0]
        self.model.touch()
        self.metadata = {"general.architecture": hy4.ARCH, "tokenizer.ggml.pre": "hy_v4", "hy_v4_flash.context_length": 1048576}
        self.reader = patch.object(prepare_module, "GGUFFile", return_value=SimpleNamespace(metadata=self.metadata))
        self.reader.start()
        self.addCleanup(self.reader.stop)
        def extract(model, output):
            (output / "tokenizer").mkdir(parents=True, exist_ok=True)
            return {}
        self.extract = patch.object(prepare_module, "extract", side_effect=extract)
        self.extract.start()
        self.addCleanup(self.extract.stop)

    def prepare(self, **kwargs):
        path = prepare_module.prepare(self.model, self.engine, self.root / "run", native_caps=CAPS, **kwargs)
        return json.loads(path.read_text())

    def test_shared_pool_is_global_and_cuda_budgets_are_native_auto(self):
        cfg = self.prepare(context=1048576, context_pool=1048576, slots=4, cuda_devices="0,1", mtp=2)
        args = cfg["args"]
        self.assertEqual(args[args.index("--context-pool") + 1], "1048576")
        self.assertEqual(args[args.index("--max-context") + 1], "1048576")
        self.assertEqual(cfg["parallel"], 4)
        self.assertNotIn("--cuda-cache-mib", args)
        self.assertNotIn("--expert-cache-mib", args)
        self.assertEqual(cfg["reasoning_effort"], "max")

    def test_wide_prefill_requires_the_native_capability_limit(self):
        caps = CAPS | {"limits": {"prefill_rows": 512}}
        path = prepare_module.prepare(self.model, self.engine, self.root / "auto-wide", native_caps=caps, cuda_devices="0")
        args = json.loads(path.read_text())["args"]
        self.assertEqual(args[args.index("--prefill-chunk") + 1], "256")
        for count in (8, 9, 256, 512):
            with self.subTest(count=count):
                path = prepare_module.prepare(self.model, self.engine, self.root / "wide", native_caps=caps,
                                              cuda_devices="0", prefill_chunk=count)
                args = json.loads(path.read_text())["args"]
                self.assertEqual(args[args.index("--prefill-chunk") + 1], str(count))
        for value in (0, 513, True, "256"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "prefill chunks"):
                prepare_module.prepare(self.model, self.engine, self.root / "bad-wide", native_caps=caps,
                                       cuda_devices="0", prefill_chunk=value)
        # The original CUDA executable did not publish limits and accepted only eight.
        with self.assertRaisesRegex(ValueError, "1..8"):
            self.prepare(cuda_devices="0", prefill_chunk=9)
        args = self.prepare(cuda_devices="0")["args"]
        self.assertEqual(args[args.index("--prefill-chunk") + 1], "8")
        for limits in ({"prefill_rows": "512"}, {"prefill_rows": True}, {"prefill_rows": 0}, []):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "limit"):
                prepare_module.prepare(self.model, self.engine, self.root / "bad-limit", native_caps=CAPS | {"limits": limits},
                                       cuda_devices="0")

    def test_cpu_reference_is_serial_f32_and_does_not_claim_cuda_features(self):
        cfg = self.prepare()
        self.assertEqual(cfg["backend"], "cpu")
        self.assertEqual(cfg["args"][cfg["args"].index("--kv-type") + 1], "f32")
        for kwargs in ({"slots": 2}, {"mtp": 2}, {"kv_type": "f16"}, {"context_pool": 16384}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(**kwargs)

    def test_feature_gates_do_not_trust_requested_config(self):
        for kwargs, message in (({"slots": 4}, "batch"), ({"mtp": 2}, "MTP"), ({"kv_location": "gpu", "cuda_devices": "0"}, "GPU KV")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                prepare_module.prepare(self.model, self.engine, self.root / "bad", native_caps=CAPS | {"features": {}}, **kwargs)

    def test_vision_uses_own_reader_and_cpu_by_default(self):
        (self.root / "strata-hy4-vision").touch()
        projector = self.root / hy4.PROJECTOR
        projector.touch()
        cfg = self.prepare(mmproj=projector, cuda_devices="0")
        self.assertFalse(cfg["vision"]["gpu"])
        self.assertEqual(Path(cfg["vision"]["exe"]).name, "strata-hy4-vision")
        self.assertIn("--mmproj", cfg["args"])
        self.assertNotIn("--vision-gpu-reserve-mib", cfg["args"])

    def test_gpu_vision_reserves_only_the_selected_encoder_card(self):
        (self.root / "strata-hy4-vision").touch()
        projector = self.root / hy4.PROJECTOR
        projector.touch()
        with self.assertRaisesRegex(ValueError, "vision memory reserve"):
            self.prepare(mmproj=projector, cuda_devices="0,1", vision_gpu=True)
        caps = CAPS | {"features": CAPS["features"] | {"vision_gpu_reserve": True}}
        path = prepare_module.prepare(self.model, self.engine, self.root / "gpu-vision", native_caps=caps,
                                      mmproj=projector, cuda_devices="0,1", vision_gpu=True)
        cfg = json.loads(path.read_text())
        self.assertTrue(cfg["vision"]["gpu"])
        args = cfg["args"]
        self.assertEqual(args[args.index("--vision-gpu-reserve-mib") + 1], "2048")

    def test_missing_native_executable_or_engram_shard_requires_repair(self):
        from strata_cli.launch.runner import files_present
        for name in hy4.FILES:
            (self.root / name).touch()
        cfg = {"arch": hy4.ARCH, "exe": str(self.engine), "args": ["--model", str(self.model)]}
        self.assertEqual(files_present(cfg), [])
        self.engine.unlink()
        (self.root / hy4.FILES[1]).unlink()
        self.assertEqual(set(files_present(cfg)), {str(self.engine), str(self.root / hy4.FILES[1])})

    def test_source_update_rebuilds_once_for_all_selected_architectures(self):
        from strata_control import hy4_install
        exe = hy4.source_tree(TOOLS.parent) / "build-hy4-cuda/src/hy4/strata-hy4"
        cfg = {"arch": hy4.ARCH, "exe": str(exe), "backend": "cuda", "gpu": [2, 3]}
        path = self.root / "strata-hy4-q4_k_m.json"
        path.write_text(json.dumps(cfg))
        fake = SimpleNamespace(ROOT=TOOLS.parent, gpu_info=Mock(side_effect=lambda i: {"arch": "86" if i == 2 else "89"}),
                               say=Mock(), warn=Mock(), write_config=Mock())
        with patch.dict(os.environ), patch.object(hy4_install, "build", return_value=(exe, CAPS)) as build, patch.object(hy4_install, "running", return_value=False):
            os.environ.pop("STRATA_HY4_ENGINE", None)
            self.assertEqual(hy4_install.update(fake, [path, path]), set())
            build.assert_called_once_with(fake, "cuda", ["86", "89"])
        self.assertEqual(fake.write_config.call_args.args[1]["native_capabilities"], CAPS)

    def test_source_update_defers_running_or_external_executable(self):
        from strata_control import hy4_install
        exe = hy4.source_tree(TOOLS.parent) / "build-hy4-cuda/src/hy4/strata-hy4"
        cfg = {"arch": hy4.ARCH, "exe": str(exe), "backend": "cuda", "gpu": [2, 3]}
        path = self.root / "strata-hy4-q4_k_m.json"
        path.write_text(json.dumps(cfg))
        fake = SimpleNamespace(ROOT=TOOLS.parent, warn=Mock())
        with patch.dict(os.environ), patch.object(hy4_install, "build") as build, patch.object(hy4_install, "running", return_value=True):
            os.environ.pop("STRATA_HY4_ENGINE", None)
            self.assertEqual(hy4_install.update(fake, [path]), {path})
            self.assertIn("deferred", fake.warn.call_args.args[0])
            build.assert_not_called()
        path.write_text(json.dumps(cfg | {"exe": str(self.engine)}))
        with patch.object(hy4_install, "build") as build:
            self.assertEqual(hy4_install.update(fake, [path]), {path})
            self.assertIn("externally managed", fake.warn.call_args.args[0])
            build.assert_not_called()

    def test_only_hy4_update_does_not_update_qwen_component(self):
        from strata_control import hy4_install
        with Home():
            setup = catalog(HW_24).strata.setup_module()
            path = self.root / "strata-hy4-q4_k_m.json"
            path.write_text(json.dumps({"arch": hy4.ARCH, "exe": str(self.engine), "args": [], "model_name": "hy4-flash"}))
            with patch.object(setup, "pip_install"), patch.object(setup, "update_installed_engine", side_effect=AssertionError("Qwen update")), patch.object(hy4_install, "update", return_value=set()) as update, patch.object(setup, "config_engine_version", return_value=()), patch.object(setup, "say"), patch.object(setup, "ok"):
                self.assertEqual(setup.update_install([path], SimpleNamespace(prebuilt="unused", build=False)), 0)
            update.assert_called_once()

    def test_source_re_setup_preserves_pool_slots_and_mtp_off(self):
        from strata_control import hy4_install
        fake = SimpleNamespace(ROOT=TOOLS.parent, OWN_PATH_FAMILIES={"hy4": hy4.family_definition("https://example.invalid/")},
                               step=Mock(), run=Mock(), pip_install=Mock(), runtime_packages=Mock(return_value=[]), download=Mock(),
                               verify_sha256=Mock(), data_folder_for=Mock(return_value=(self.root, [])), home=Mock(return_value=self.root),
                               explicit_args=Mock(return_value=[]), write_setup_config=Mock(), write_run_script=Mock(return_value=self.root / "start.sh"), ok=Mock())
        for name in hy4.FILES:
            (self.root / name).touch()
        old = {"arch": hy4.ARCH, "backend": "cuda", "gpu": [2,3], "parallel": 4, "context_pool": 1048576,
               "mtp": False, "drafting": "off", "args": ["--max-context", "1048576", "--context-pool", "1048576", "--slots", "4", "--mtp", "0"]}
        (self.root / "strata-hy4-q4_k_m.json").write_text(json.dumps(old))
        a = SimpleNamespace(model="Q4_K_M", backend="cuda", drafter=None, kv=None, gguf_dir=str(self.root), models_dir=str(self.root),
                            context=None, context_pool=None, parallel=None, mtp=None, vision="none", no_start=True,
                            vram_reserve_mib=None, host=None, api_key=None, browser=False)
        with patch.dict(os.environ), patch.object(hy4_install, "build", return_value=(self.engine, CAPS)):
            os.environ.pop("STRATA_HY4_ENGINE", None)
            self.assertEqual(hy4_install.install(fake, a, gpu={"index":2,"archs":["86"]}, multi=[2,3]), 0)
            cfg = fake.write_setup_config.call_args.args[1]
            self.assertEqual((cfg["parallel"],cfg["context_pool"],cfg["mtp"]), (4,1048576,False))
            a.context, a.context_pool, a.parallel, a.mtp = 8192, 8192, 1, "on"
            self.assertEqual(hy4_install.install(fake, a, gpu={"index":2,"archs":["86"]}, multi=[2,3]), 0)
            cfg = fake.write_setup_config.call_args.args[1]
            self.assertEqual((cfg["parallel"],cfg["context_pool"],cfg["mtp"]), (1,8192,True))

    def test_installer_uses_owned_source_build_then_both_shards(self):
        from strata_control import hy4_install
        fake = SimpleNamespace(ROOT=TOOLS.parent, OWN_PATH_FAMILIES={"hy4": hy4.family_definition("https://example.invalid/")},
                               step=Mock(), run=Mock(), pip_install=Mock(), runtime_packages=Mock(return_value=[]), download=Mock(),
                               verify_sha256=Mock(), data_folder_for=Mock(return_value=(self.root, [])), home=Mock(return_value=self.root),
                               explicit_args=Mock(return_value=[]), write_setup_config=Mock(), write_run_script=Mock(return_value=self.root / "start.sh"), ok=Mock())
        for name in hy4.FILES:
            (self.root / name).touch()
        a = SimpleNamespace(model="Q4_K_M", backend="cuda", drafter=None, kv=None, gguf_dir=str(self.root), models_dir=str(self.root),
                            context=8192, context_pool=8192, parallel=1, mtp="off", vision="none", no_start=True,
                            vram_reserve_mib=1024, host=None, api_key=None, browser=False)
        with patch.dict(os.environ, {}, clear=False), patch.object(hy4, "engine_path", return_value=self.engine), patch.object(hy4, "capabilities", return_value=CAPS), patch.object(hy4_install, "running", return_value=False):
            os.environ.pop("STRATA_HY4_ENGINE", None)
            self.assertEqual(hy4_install.install(fake, a, gpu={"index": 2, "archs": ["86"]}, multi=[2, 3]), 0)
        self.assertEqual(fake.run.call_args.args[0][1], str(TOOLS / "hy4_build.py"))
        self.assertEqual([c.args[0].name for c in fake.verify_sha256.call_args_list], list(hy4.FILES))
        cfg = fake.write_setup_config.call_args.args[1]
        self.assertEqual(cfg["gpu"], [2, 3])
        self.assertEqual(cfg["args"][cfg["args"].index("--cuda-devices") + 1], "0,1")
        fake.download.assert_not_called()
        with acquisition.forbidden(), self.assertRaises(acquisition.AcquisitionRefused):
            acquisition.require("building Hy4")


if __name__ == "__main__":
    unittest.main()
