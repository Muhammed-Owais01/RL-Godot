import os
import subprocess
import time
from typing import List


class ProcessManager:
    def __init__(self, godot_exe: str, project_path: str, base_port: int = 5005):
        self.godot_exe = godot_exe
        self.project_path = project_path
        self.base_port = base_port
        self.processes: List[subprocess.Popen] = []

    def start(self, count: int, startup_delay_sec: float = 0.5, show_window: bool = False) -> List[int]:
        ports = []
        for i in range(count):
            port = self.base_port + i
            env = os.environ.copy()

            args = [
                self.godot_exe,
                "--path",
                self.project_path,
                f"--port={port}",
            ]

            if not show_window:
                args.insert(1, "--headless")

            proc = subprocess.Popen(args, env=env)
            self.processes.append(proc)
            ports.append(port)

            time.sleep(startup_delay_sec)
            if proc.poll() is not None:
                raise RuntimeError(f"Godot exited early on port {port}")
        return ports

    def stop_all(self) -> None:
        for proc in self.processes:
            proc.terminate()
        self.processes.clear()
