from pathlib import Path

import tomlkit as toml
import tomlkit


class Config:
    def __init__(self):
        self._file = Path("config.toml")

    def get_instance(self):
        with self._file.open() as f:
            toml_dict = toml.parse(f.read())
        f.close()
        return toml_dict

    def change_value(self, key, subkey, value):
        current = self.get_instance()
        current[key][subkey] = value
        with self._file.open("w") as f:
            f.write(tomlkit.dumps(current))
        f.close()
