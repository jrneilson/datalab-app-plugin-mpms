from datalab_app_plugin_mpms import __version__
from datalab_app_plugin_mpms.blocks import ExampleDataBlock


def test_version():
    assert __version__


def test_example_data_block():
    block = ExampleDataBlock
    assert block.version == __version__
