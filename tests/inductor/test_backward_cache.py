# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the AOTAutograd backward FxGraph layout bridge (PR #3051).

The compiled backward reuses the forward's committed saved-activation layouts
through a ``graph_id``-keyed registry.  That bridge only works if a single
``graph_id`` is used consistently at all three sites that touch the registry
across the forward and backward compiles of one AOT graph:

  1. forward deposit  -- capture_forward_output_layouts, key graph.graph_id
  2. backward consume -- _consume_forward_output_layouts, key graph.graph_id
  3. backward key     -- forward_output_layouts_for(fx_kwargs["graph_id"])

Today torch's ``compile_fx`` threads one ``graph_id`` to both compilers, so the
three agree -- but nothing in-tree pins that.  If a torch upgrade ever diverged
them, the backward would silently mis-key the disk cache or drop the layout
reuse and produce *wrong gradients with no error*.  ``test_graph_id_three_way_
contract`` fails loudly if the contract shifts (review comment from @mudhakar on
PR #3051); ``test_backward_cache_key_folds_in_forward_layouts`` is a fast
hash-level guard that site 3 actually consumes the ``graph_id``-keyed registry.

The instrumentation monkey-patches the three sites at runtime only (restored on
teardown); no ``torch_spyre`` source is modified.
"""

import unittest

import torch

import torch_spyre  # noqa: F401  registers "spyre" + installs the cache-key patch
import torch._inductor.config as t_inductor_config
import torch_spyre._inductor.passes as _passes
import torch_spyre._inductor.propagate_layouts as _pl
from torch._inductor.codecache import FxGraphCachePickler, FxGraphHashDetails
from torch._inductor.utils import fresh_cache

SPYRE = torch.device("spyre")
DT = torch.float16


def _net(x, w1, w2):
    # Elementwise-only "MLP": the two sigmoid outputs are saved activations that
    # flow into the backward graph, so the backward has saved-activation inputs
    # whose layouts come through the registry bridge.  No matmul / broadcast
    # (those are separate backend gaps unrelated to this contract).
    h = torch.sigmoid(x * w1)
    return torch.sigmoid(h * w2)


class TestBackwardGraphIdContract(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)
        # Isolate from earlier compiles in this process: dynamo state + the
        # in-process forward/tangent layout registries.  (The absolute graph_id
        # counter is process-global and NOT reset here -- which is exactly why
        # the assertions below never reference an absolute graph_id value.)
        torch._dynamo.reset()
        _pl._forward_output_layouts.clear()
        _pl._backward_tangent_layouts.clear()

    def _instrument(self):
        """Wrap the three ``graph_id`` sites; return a ``seen`` dict they populate.

        Restored on teardown via ``addCleanup``.  ``capture_forward_output_layouts``
        is embedded into the pre-scheduling pass list from the ``passes.py`` binding
        (``passes.py`` imported it by name), so THAT binding must be patched too --
        patching only the ``propagate_layouts`` attribute is a silent no-op for the
        forward-capture pass.  ``_consume_forward_output_layouts`` and
        ``forward_output_layouts_for`` are resolved by name at call time, so
        patching them on ``propagate_layouts`` is sufficient.  The tangent guard
        (System 2) is instrumented the same way: ``capture_backward_tangent_layouts``
        via the ``passes`` binding, ``pop_backward_tangent_layouts`` via
        ``propagate_layouts`` (re-imported inside the guard installer).
        """
        seen = {"fwd_deposited_gids": set()}

        orig_cap = _pl.capture_forward_output_layouts
        orig_consume = _pl._consume_forward_output_layouts
        orig_lookup = _pl.forward_output_layouts_for

        def cap(graph):
            r = orig_cap(graph)
            if not graph.is_backward:
                seen["capture_ran"] = True
                gid = graph.graph_id
                # Record only graph_ids that actually DEPOSITED a non-empty entry
                # (a forward with no tiled outputs deposits nothing).  Checked
                # right after the deposit, before the backward consume pops it.
                if gid is not None and gid in _pl._forward_output_layouts:
                    seen["fwd_deposited_gids"].add(gid)
            return r

        def consume(graph):
            r = orig_consume(graph)  # POPS the registry entry -- call exactly once
            seen["bwd_consume_gid"] = graph.graph_id
            seen["bwd_consume_nonempty"] = bool(r)
            return r

        def lookup(graph_id):  # arg is the graph_id int from fx_kwargs, not a graph
            seen["cachekey_gid"] = graph_id
            return orig_lookup(graph_id)

        # --- System 2: the tangent guard's OWN graph_id round-trip ---
        # capture_backward_tangent_layouts deposits under graph.graph_id, but
        # _install_backward_tangent_guard consumes via
        # pop_backward_tangent_layouts(compiler_config_extra.graph_id) -- a fourth
        # graph_id source.  The capture pass is in the pass list (patch the _passes
        # binding); pop is re-imported by name inside the installer (patch the
        # propagate_layouts attribute).
        orig_cap_bw = _pl.capture_backward_tangent_layouts
        orig_pop = _pl.pop_backward_tangent_layouts

        def cap_bw(graph):
            r = orig_cap_bw(graph)
            if graph.is_backward:
                seen["tangent_deposit_gid"] = graph.graph_id
            return r

        def pop_tangent(graph_id):  # graph_id == compiler_config_extra.graph_id
            seen["tangent_guard_gid"] = graph_id
            return orig_pop(graph_id)

        _passes.capture_forward_output_layouts = cap  # load-bearing (pass-list binding)
        _pl.capture_forward_output_layouts = cap  # mirror; not load-bearing
        _pl._consume_forward_output_layouts = consume
        _pl.forward_output_layouts_for = lookup
        _passes.capture_backward_tangent_layouts = cap_bw  # load-bearing (pass list)
        _pl.capture_backward_tangent_layouts = cap_bw  # mirror
        _pl.pop_backward_tangent_layouts = pop_tangent

        def restore():
            _passes.capture_forward_output_layouts = orig_cap
            _pl.capture_forward_output_layouts = orig_cap
            _pl._consume_forward_output_layouts = orig_consume
            _pl.forward_output_layouts_for = orig_lookup
            _passes.capture_backward_tangent_layouts = orig_cap_bw
            _pl.capture_backward_tangent_layouts = orig_cap_bw
            _pl.pop_backward_tangent_layouts = orig_pop

        self.addCleanup(restore)
        return seen

    def test_graph_id_three_way_contract(self):
        seen = self._instrument()

        # Cache ON + a fresh dir -> the forward compiles fresh (deposit runs) and
        # the backward compiles fresh (consume + cache-key both run on the miss).
        compiled = torch.compile(_net, backend="inductor", dynamic=False)
        with (
            t_inductor_config.patch(
                {"force_disable_caches": False, "fx_graph_cache": True}
            ),
            fresh_cache(),
        ):
            xs = torch.randn(8, 64, dtype=DT)
            w1s = torch.randn(8, 64, dtype=DT)
            w2s = torch.randn(8, 64, dtype=DT)

            def leaves(dev, dtype):
                return [
                    t.to(device=dev, dtype=dtype).detach().requires_grad_(True)
                    for t in (xs, w1s, w2s)
                ]

            x, w1, w2 = leaves(SPYRE, DT)
            xc, w1c, w2c = leaves("cpu", torch.float32)

            compiled(x, w1, w2).sum().backward()
            _net(xc, w1c, w2c).sum().backward()

        # Precondition (guard, do not assert-pass): if the forward was served from
        # the compile cache, the capture pass never ran and the contract is
        # vacuous.  Should not happen under fresh_cache()+dynamo.reset(), but a
        # warm cache must skip, never report a false green.
        if not seen.get("capture_ran"):
            self.skipTest("forward served from compile cache; contract not exercised")

        gid_consume = seen.get("bwd_consume_gid")
        gid_key = seen.get("cachekey_gid")

        # A1 -- the two BACKWARD sites use the same graph_id.  If a torch bump makes
        # graph.graph_id (consume) and fx_kwargs["graph_id"] (cache key) diverge,
        # the key is computed against a different layout set than propagation uses.
        self.assertIsNotNone(
            gid_consume, "backward consume did not run (graph_id=None)"
        )
        self.assertIsNotNone(gid_key, "backward cache-key patch did not run")
        self.assertEqual(
            gid_consume,
            gid_key,
            f"backward graph_id diverged across sites: consume={gid_consume} "
            f"cache_key={gid_key}. compile_fx no longer threads one graph_id to "
            f"both backward sites -- the cache would mis-key.",
        )

        # A2 -- the forward DEPOSITED under the graph_id the backward withdraws
        # from.  Set membership, never equality to a single captured value:
        # torch.compile can emit multiple forward variants and the backward binds
        # to whichever produced its saved activations.
        self.assertIn(
            gid_consume,
            seen["fwd_deposited_gids"],
            f"backward consumed graph_id={gid_consume} but the forward only "
            f"deposited under {sorted(seen['fwd_deposited_gids'])}; the fwd/bwd "
            f"graph_id link is broken -- layout reuse would silently drop.",
        )

        # A3 -- the withdrawal actually returned layouts, not an empty fallback.
        self.assertTrue(
            seen.get("bwd_consume_nonempty"),
            "backward consumed an EMPTY layout set for a graph with saved "
            "activations -- the bridge silently fell back to IR defaults.",
        )

        # A4 -- System 2 (the tangent guard) keys off the SAME graph_id.  Its
        # assumed-layout note is deposited by capture_backward_tangent_layouts under
        # graph.graph_id, but consumed by _install_backward_tangent_guard via
        # pop_backward_tangent_layouts(compiler_config_extra.graph_id) -- a FOURTH
        # graph_id source.  If it diverges (e.g. a torch bump renames the attr so
        # getattr(..., "graph_id", None) returns None), pop returns {}, the installer
        # bails at `if not tangent_layouts: return`, the guard is silently NOT
        # installed, and a non-default tangent goes back to silently-wrong gradients
        # -- the exact failure the guard exists to prevent.
        gid_tangent_deposit = seen.get("tangent_deposit_gid")
        gid_tangent_guard = seen.get("tangent_guard_gid")
        self.assertIsNotNone(
            gid_tangent_deposit, "capture_backward_tangent_layouts did not run"
        )
        self.assertIsNotNone(
            gid_tangent_guard,
            "tangent guard read graph_id=None (compiler_config_extra.graph_id "
            "missing) -- the guard would silently not install",
        )
        self.assertEqual(
            gid_tangent_guard,
            gid_tangent_deposit,
            f"tangent-guard graph_id diverged: deposit (graph.graph_id)="
            f"{gid_tangent_deposit} vs consume (compiler_config_extra.graph_id)="
            f"{gid_tangent_guard}; pop would return {{}} and the guard would "
            f"silently not install -> non-default tangents silently wrong.",
        )
        self.assertEqual(
            gid_tangent_guard,
            gid_consume,
            f"tangent-guard graph_id {gid_tangent_guard} != backward graph_id "
            f"{gid_consume} used by the rest of the chain.",
        )

        # A5 -- and the resulting gradients are numerically correct.  A mis-keyed
        # cache or dropped layout reuse would corrupt these with no error.
        for name, g, gc in (
            ("x", x.grad, xc.grad),
            ("w1", w1.grad, w1c.grad),
            ("w2", w2.grad, w2c.grad),
        ):
            self.assertIsNotNone(g, f"{name}.grad is None")
            torch.testing.assert_close(g.cpu().float(), gc, atol=3e-2, rtol=3e-2)

    def test_backward_cache_key_folds_in_forward_layouts(self):
        # Hash-level guard that the cache-key site (3) actually consumes the
        # graph_id-keyed registry: two different forward saved-activation layouts
        # registered under the same graph_id must yield DIFFERENT backward keys
        # (else two differently-tiled backwards collide and reuse each other's
        # codegen), and the same layout must be deterministic.  No compile needed.
        stl_a = torch.randn(64, 128, device=SPYRE, dtype=DT).device_tensor_layout()
        stl_b = (
            torch.randn(64, 128, device=SPYRE, dtype=DT)
            .transpose(0, 1)
            .contiguous()
            .device_tensor_layout()
        )
        self.assertNotEqual(repr(stl_a), repr(stl_b))

        def bw(sigmoid, tangents_1):
            return sigmoid * tangents_1

        gm = torch.fx.symbolic_trace(bw)
        example_inputs = [torch.randn(64, 128), torch.randn(64, 128)]
        gid = 777
        fx_kwargs = {"is_backward": True, "graph_id": gid}

        def key_for(stl):
            _pl._forward_output_layouts.clear()
            _pl._forward_output_layouts[gid] = {"sigmoid": stl}
            details = FxGraphHashDetails(gm, list(example_inputs), fx_kwargs, [])
            return FxGraphCachePickler(gm).get_hash(details)

        try:
            hash_a = key_for(stl_a)
            hash_b = key_for(stl_b)
            hash_a2 = key_for(stl_a)
        finally:
            _pl._forward_output_layouts.clear()

        self.assertNotEqual(
            hash_a,
            hash_b,
            "different forward saved-activation layouts collided to the same "
            "backward cache key",
        )
        self.assertEqual(
            hash_a, hash_a2, "same forward layout produced non-deterministic keys"
        )


if __name__ == "__main__":
    unittest.main()
