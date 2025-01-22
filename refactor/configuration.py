import abc
import dataclasses
import pathlib
import yaml


@dataclasses.dataclass
class Configuration(abc.ABC):

    def __init__(self, config_file: pathlib.Path = None):
        super().__init__()

        if config_file is not None:
            self.load(config_file)

    #     self.type_check()

    # def type_check(self, data: dict = None):
    #     config = data.__dict__ if data is not None else self.__dict__
    #     data_class = data.__class__ if data is not None else self.__class__

    #     for key, value in config.items():
    #         field_type = data_class.__annotations__.get(key)
    #         if field_type is not None:
    #             if dataclasses.is_dataclass(field_type):
    #                 self.type_check(data=value)
    #             if not isinstance(value, field_type):
    #                 raise TypeError(f"Expected type {field_type} for {key}, but got {type(value)}.")

    def load(self, config_file: pathlib.Path):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        for key, value in config.items():
            field_type = self.__class__.__annotations__.get(key)
            if field_type:
                value = self._process_field(value, field_type)
                setattr(self, key, value)

    def _process_field(self, data, field_type):
        if not isinstance(data, dict):
            return data

        processed_data = {}
        for key, val in data.items():
            subfield_type = field_type.__annotations__.get(key)
            processed_data[key] = self._process_field(val, subfield_type)

        return field_type(**processed_data)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
