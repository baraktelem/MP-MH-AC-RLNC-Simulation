class GeneralUnit:
    def __init__(self, unit_name: str = "GeneralUnit", t: int = 0, debug: bool = False):
        self.name = unit_name
        self.t = t
        self._debug = debug

    def run_step(self, time: int = None):
        self.t = time if time is not None else self.t + 1

    def sim_print(self, message: str, append_message: bool = False):
        if self._debug:
            print(f"[{self.t}] {self.name}{"." if append_message else ": "}{message}")

