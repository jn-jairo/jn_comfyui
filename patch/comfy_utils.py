from comfy.utils import *
import comfy.utils

import inspect
import os
import re

import time
import psutil
import GPUtil
import platform
import subprocess

def patchComfyUtils(config):
    def get_extension_calling():
        for frame in inspect.stack():
            if os.sep + "custom_nodes" + os.sep in frame.filename \
                and os.sep + "patch" + os.sep + "comfy_utils.py" not in frame.filename \
                and os.sep + "patch" + os.sep + "comfy_model_management.py" not in frame.filename:
                stack_module = inspect.getmodule(frame[0])
                if stack_module:
                    stack = []

                    parts = re.sub(r".*\.?custom_nodes\.([^\.]+).*", r"\1", stack_module.__name__.replace(os.sep, ".")).split(".")

                    while len(parts) > 0:
                        stack.append(".".join(parts))
                        parts.pop()

                    return stack

        return None

    def clear_line(n=1):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        for i in range(n):
            print(LINE_UP, end=LINE_CLEAR)

    def func_sleep(seconds, pbar=None):
        while seconds > 0:
            print(f"Sleeping {seconds} seconds")
            time.sleep(1)
            seconds -= 1
            clear_line()
            if pbar is not None:
                pbar.update(1)

    def get_processor_name():
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command ="sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).strip()
        return ""

    def get_temperatures():
        temperatures = []

        if platform.system() == "Linux":
            cpu_max_temp = 0

            for k, v in psutil.sensors_temperatures(fahrenheit=False).items():
                for t in v:
                    if t.current > cpu_max_temp:
                        cpu_max_temp = t.current

            temperatures.append({
                "label": get_processor_name(),
                "temperature": cpu_max_temp,
                "kind": "CPU",
            })

        for gpu in GPUtil.getGPUs():
            temperatures.append({
                "label": gpu.name,
                "temperature": gpu.temperature,
                "kind": "GPU",
            })

        return temperatures

    cooldown = {
        "waiting": False,
    }

    def _wait_cooldown(max_temperature=70, safe_temperature=60, seconds=2, max_seconds=0):
        if cooldown["waiting"]:
            return

        cooldown["waiting"] = True

        try:
            max_temperature, safe_temperature = max(max_temperature, safe_temperature), min(max_temperature, safe_temperature)

            if max_temperature <= 0:
                return

            if safe_temperature <= 0:
                safe_temperature = max_temperature

            if max_seconds == 0:
                max_seconds = 0xffffffffffffffff

            seconds = max(1, seconds)
            max_seconds = max(seconds, max_seconds)
            times = max_seconds // seconds

            hot = True

            # Start with the max temperature, so if not above it don't cool down.
            limit_temperature = max_temperature

            while hot and times > 0:
                temperatures = [f"{t['kind']} {t['label']}: {t['temperature']}" for t in get_temperatures() if t["temperature"] > limit_temperature]
                hot = len(temperatures) > 0

                if hot:
                    # Switch to safe temperature to cool down to that temperature
                    limit_temperature = safe_temperature
                    print(f"Too hot! Limit temperature: [ {limit_temperature} ] Current temperature: [ " + " | ".join(temperatures) + " ]")
                    pbar = ProgressBar(seconds)
                    func_sleep(seconds, pbar)
                    clear_line()
                    times -= 1
        finally:
            cooldown["waiting"] = False

    def wait_cooldown(kind="execution", max_temperature=None, safe_temperature=None, seconds=None, max_seconds=None):
        max_temperature = max_temperature if max_temperature is not None else config["temperature"]["limit"][kind]["max"]
        safe_temperature = safe_temperature if safe_temperature is not None else config["temperature"]["limit"][kind]["safe"]
        cool_down_each = seconds if seconds is not None else config["temperature"]["cool_down"]["each"]
        cool_down_max = max_seconds if max_seconds is not None else config["temperature"]["cool_down"]["max"]

        if safe_temperature > 0:
            _wait_cooldown(
                max_temperature=max_temperature,
                safe_temperature=safe_temperature,
                seconds=cool_down_each,
                max_seconds=cool_down_max,
            )

    comfy.utils.get_extension_calling = get_extension_calling
    comfy.utils.clear_line = clear_line
    comfy.utils.func_sleep = func_sleep
    comfy.utils.wait_cooldown = wait_cooldown

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        self.current = value
        if self.hook is not None:
            self.hook(self.current, self.total, preview)
        comfy.utils.wait_cooldown(kind="progress")

    comfy.utils.ProgressBar.update_absolute = update_absolute

PATCHES = {
    "20_comfy_utils": patchComfyUtils,
}
