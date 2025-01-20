from importlib import resources


_data_ref = resources.files("openadmet_toolkit/data")

example_vendor_library = (_data_ref / "truncated_vendor_library.csv").as_posix()


