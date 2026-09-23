import time

from capability_anatomy.protocols import CORE_API_VERSION
from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter


class ExternalModelPlugin(SyntheticModelAdapter):
    name = "external.fixture-model"


class ExternalRuntimePlugin:
    name = "external.fixture-runtime"
    version = "1"
    api_version = CORE_API_VERSION
    capabilities = frozenset({
        "runtime.execute", "runtime.memory", "runtime.synchronization",
        "runtime.timing", "runtime.tokens", "runtime.fixture.remote",
    })
    calls = {name: 0 for name in ("execute", "synchronize", "memory_bytes", "token_counts", "clock")}

    def execute(self, adapter, loaded, request):
        self.calls["execute"] += 1
        return adapter.execute(loaded, request)

    def synchronize(self, adapter, loaded):
        self.calls["synchronize"] += 1
        adapter.synchronize(loaded)

    def memory_bytes(self, adapter, loaded):
        self.calls["memory_bytes"] += 1
        return adapter.memory_bytes(loaded)

    def token_counts(self, adapter, loaded, request, result):
        self.calls["token_counts"] += 1
        return adapter.token_counts(loaded, request, result)

    def clock(self):
        self.calls["clock"] += 1
        return time.perf_counter()
