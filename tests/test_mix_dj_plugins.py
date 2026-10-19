'''test_mix_dj_plugins.py
Unit tests for the mix-DJ placeholder plugins.
''' 
import importlib
import pathlib

# Helper to import a module from the plugins directory
def import_plugin(module_name: str):
    plugins_path = pathlib.Path(__file__).resolve().parents[1] / "plugins" / "mix_dj"
    spec = importlib.util.spec_from_file_location(module_name, plugins_path / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_mpc_sampler_register():
    mpc = import_plugin("mpc_sampler")
    meta = mpc.register_plugin()
    assert meta["name"] == "mpc_sampler"
    assert hasattr(mpc, "MPCSampler")
    instance = mpc.MPCSampler()
    assert instance.sample("test") == "test"

def test_dj_producer_tool_register():
    dj = import_plugin("dj_producer_tool")
    meta = dj.register_plugin()
    assert meta["name"] == "dj_producer_tool"
    assert hasattr(dj, "DJProducerTool")
    instance = dj.DJProducerTool()
    assert instance.process("track") == "track"

def test_seratonizer_register():
    ser = import_plugin("seratonizer")
    meta = ser.register_plugin()
    assert meta["name"] == "seratonizer"
    assert hasattr(ser, "Seratonizer")
    instance = ser.Seratonizer()
    assert instance.convert("audio") == "audio"
