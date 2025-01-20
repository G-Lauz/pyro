import abc

import numpy


class PostInitializationMetaclass(abc.ABCMeta):
    """
    Add a hook being called after the initialization of the class and all its subclasses.
    """
    def __call__(self, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        
        if hasattr(instance, "__post_init__"):
            instance.__post_init__()

        return instance


class Configuration(abc.ABC, metaclass=PostInitializationMetaclass):
    """
    A configuration class for mechanical systems.

    The configuration is a dictionary of the form:
    
    ```JSON
    {
        "q": {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0
        },
        "dq": {
            "dx": 0.0,
            "dy": 0.0,
            "dtheta": 0.0
        }
    }
    ```

    The configuration can be of any order. The order is the number of derivatives of the configuration that are stored dynamically.
    This mean that you only need to define the first order of the configuration and the rest will be created dynamically.

    Examples:
    ```Python
    class ConcreteConfiguration(Configuration):
        x: float
        y: float
        theta: float

    config = ConcreteConfiguration(order=2)
    
    # Access the derived configuration
    print(config.q) # {"x": 0.0, "y": 0.0, "theta": 0.0}
    print(config.dq) # {"dx": 0.0, "dy": 0.0, "dtheta": 0.0}
    print(config.["dq"]) # {"dx": 0.0, "dy": 0.0, "dtheta": 0.0}

    # Access a configuration value
    print(config.x) # 0.0
    print(config.dx) # 0.0
    print(config["dtheta"]) # 0.0

    # Initialize the configuration with a state vector
    config = ConcreteConfiguration(order=2, initial_state=numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    print(config.q) # {"x": 1.0, "y": 2.0, "theta": 3.0}
    print(config.dq) # {"dx": 4.0, "dy": 5.0, "dtheta": 6.0}

    # Convert the configuration to a state vector
    vector, labels = config.to_states()
    print(vector) # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    print(labels) # ["x", "y", "theta", "dx", "dy", "dtheta"]

    # Convert a state vector to a configuration
    config = ConcreteConfiguration(order=2)
    config.from_states(numpy.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    print(config.q) # {"x": 10.0, "y": 20.0, "theta": 30.0}
    print(config.dq) # {"dx": 40.0, "dy": 50.0, "dtheta": 60.0}

    # Get the length of the configuration
    print(len(config)) # 3

    # Get the length of the state vector
    print(config.state_length()) # 6
    ```
    """

    config: dict
   
    def __init__(self, order = 1, initial_state: numpy.ndarray = None):
        super().__init__()

        if order < 1:
            raise ValueError("The order of the configuration must be greater than 0.")

        self.order = order
        self.initial_state = initial_state

    def __post_init__(self):
        annotations = self.__annotations__
        self.config = {
            "q": annotations
        }

        for key, _ in self.config["q"].items():
            self.config["q"][key] = 0.0

        for i in range(1, self.order):
            derivative = "d"*i + "q"
            self.config[derivative] = {}
            for key in self.config["q"]:
                key = "d"*i + key
                self.config[derivative][key] = 0.0

        if self.initial_state is not None:
            self.from_states(self.initial_state)

    def __getattr__(self, name):
        derivative_order = len(name) - len(name.lstrip("d"))
        derivative_group = "d"*derivative_order + "q"

        if derivative_order + 1 > self.order:
            tips = f"You requested a derivative of order {derivative_order + 1} but the configuration is only of order {self.order}."

            raise AttributeError(f"{self.__class__.__name__} has no attribute `{name}`. {tips}")

        if name == derivative_group:
            return self.config[derivative_group]

        try:
            return self.config[derivative_group][name]
        except KeyError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute `{name}`.")
        
    def __setattr__(self, name, value):
        if "config" not in self.__dict__:
            super().__setattr__(name, value)
            return

        derivative_order = len(name) - len(name.lstrip("d"))
        derivative_group = "d" * derivative_order + "q"

        if derivative_order + 1 > self.__dict__.get("order", 1):
            raise AttributeError(
                f"You cannot assign to a derivative of order {derivative_order + 1} "
                f"when the configuration order is {self.__dict__.get('order', 1)}."
            )

        if name == derivative_group:
            if not isinstance(value, dict):
                raise ValueError(f"Expected a dictionary for {name}, got {type(value)}.")
            self.config[derivative_group].update(value)
            return

        if derivative_group in self.config and name in self.config[derivative_group]:
            self.config[derivative_group][name] = value
            return

        super().__setattr__(name, value)


    def __getitem__(self, name):
        return self.__getattr__(name)
    
    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __len__(self):
        return len(self.config["q"])
    
    def state_length(self):
        return len(self) * (self.order)

    def to_states(self):
        """
        Convert the configuration to a state vector.

        :return: The state vector and the labels. (n_states, 1), (n_states, 1)
        """
        vector = []
        labels = []
        for key in self.config.keys():
            for subkey in self.config[key].keys():
                vector.append(self.config[key][subkey])
                labels.append(subkey)
        
        return numpy.array(vector), labels

    def from_states(self, vector):
        """
        Convert a state vector to a configuration.

        :param vector: The state vector. (n_states, 1)
        """
        idx = 0
        for key in self.config.keys():
            for subkey in self.config[key].keys():
                self.config[key][subkey] = vector[idx]
                idx += 1


if __name__ == "__main__":
    class ConcreteConfiguration(Configuration):
        x: float
        y: float
        theta: float

    config = ConcreteConfiguration(order=2)
    print(config.config)

    config = ConcreteConfiguration(order=2, initial_state=numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    print(config.config)
    print(config.to_states())

    config = ConcreteConfiguration(order=2)
    config.from_states(numpy.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    print(config.config)

    print(config.q)
    print(config.dx)
    print(config["dq"])
    print(config["theta"])

    try:
        print(config.ddq)
    except AttributeError as e:
        print(f"Correctly raised an AttributeError: \"{e}\"")

    try:
        print(config.ddx)
    except AttributeError as e:
        print(f"Correctly raised an AttributeError: \"{e}\"")

    try:
        print(config.do_not_exist)
    except AttributeError as e:
        print(f"Correctly raised an AttributeError: \"{e}\"")

    config.dq = {"dx": 1.0, "dy": 2.0, "dtheta": 3.0}
    print(config.config)

    config.dx = 42.0
    print(config.config)

    config["dq"] = {"dx": 2.0, "dy": 4.0, "dtheta": 6.0}
    print(config.config)

    config["dx"] = 20.0
    print(config.config)

    print(len(config))
    print(config.state_length())

    config = ConcreteConfiguration(order=3)
    print(config.config)

    config = ConcreteConfiguration()
    print(config.config)
