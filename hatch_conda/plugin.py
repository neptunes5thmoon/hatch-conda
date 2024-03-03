from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Callable

import pexpect
from hatch.env.plugin.interface import EnvironmentInterface


class ShellManager:
    def __init__(self, environment: EnvironmentInterface):
        self.environment = environment

    def enter_bash(self, path: str, args: list[str], cmdl: str) -> None:
        self.spawn_linux_shell(path or "bash", args or ["-i"], cmdl)

    def enter_zsh(self, path: str, args: list[str], cmdl: str) -> None:
        self.spawn_linux_shell(path or "zsh", args or ["-i"], cmdl)
        
    def enter_fish(self, path: str, args: list[str], cmdl: str) -> None:
        self.spawn_linux_shell(path or "fish", args or ["-i"], cmdl)

    def spawn_linux_shell(
        self, path: str, args: list[str] | None = None, cmdl: str = "", callback: Callable | None = None
    ) -> None:
        columns, lines = shutil.get_terminal_size()
        terminal = pexpect.spawn(path, args=args, dimensions=(lines, columns))

        def sigwinch_passthrough(sig: int, data: FrameType | None) -> None:
            new_columns, new_lines = shutil.get_terminal_size()
            terminal.setwinsize(new_lines, new_columns)

        signal.signal(signal.SIGWINCH, sigwinch_passthrough)  # type: ignore

        terminal.sendline(cmdl)

        if callback is not None:
            callback(terminal)

        terminal.interact(escape_character=None)
        terminal.close()

        self.environment.platform.exit_with_code(terminal.exitstatus)


class CondaEnvironment(EnvironmentInterface):
    PLUGIN_NAME = "conda"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._config_command = None
        self._config_conda_forge = None
        self._config_environment_file = None
        self._config_prefix = None
        self.__python_version = None
        self._config_use_default_conda_env_path = None
        self._access_cl_option = None
        # f"{self.metadata.core.name}_{self.name}_{self.python_version}"
        #self.conda_env_name = None
        self.project_path = "."
        self.shells = ShellManager(self)

    @staticmethod
    def get_option_types():
        return {
            "command": str,
            "conda-forge": bool,
            "environment-file": str,
            "use_default_conda_env_path": bool,
            "prefix": (str, type(None)),
        }

    def _config_value(self, field_name, default, valid=None):
        class_name = f'_config_{field_name.replace("-", "_")}'
        if self.__dict__[class_name] is None:
            value = self.config.get(field_name, default)
            if not isinstance(value, self.get_option_types()[field_name]):
                raise TypeError(
                    f"Field `tool.hatch.envs.{self.name}.{field_name}` must be a "
                    + "`{self.get_option_types()[field_name]}`"
                )
            if valid is not None and value not in valid:
                raise ValueError(
                    f"Field `tool.hatch.envs.{self.name}.{field_name}` must be any of [{valid}] values.")
            self.__dict__[class_name] = value
        return self.__dict__[class_name]

    @property
    def config_command(self):
        return self._config_value("command", "conda", ["conda", "mamba", "micromamba"])

    @property
    def config_conda_forge(self):
        return self._config_value("conda-forge", True)

    @property
    def config_prefix(self):
        return self._config_value("prefix", None)

    @property
    def environment_file(self):
        return self._config_value("environment-file", "")

    @property
    def config_use_default_conda_env_path(self):
        return self._config_value('use_default_conda_env_path', True, valid=[False, True])

    @property
    def python_version(self):
        if self.__python_version is None:
            python_version = self.config.get("python", "")
            if not python_version:
                python_version = ".".join(map(str, sys.version_info[:2]))
            elif python_version.isdigit() and len(python_version) > 1:
                python_version = f"{python_version[0]}.{python_version[1:]}"

            self.__python_version = python_version

        return self.__python_version

    def _get_conda_env_path(self, name: str):
        if self.config_prefix is not None:
            return self.config_prefix

        if self.config_command == "micromamba":
            output = self.platform.check_command_output([self.config_command, "info", "--name", name])

            match_env_location = r"env location : ([\S]*)\n"
            return re.findall(match_env_location, output)[0]

        else:
            return os.path.join(self.data_directory, f'{self.metadata.core.name}_{self.name}_{self.python_version}')

    def _get_conda_env_path(self, name):
        output = json.loads(self.platform.check_command_output(
            [self.config_command, 'info', '--json']))
        envs_dirs = output["envs_dirs"]
        for env_dir in envs_dirs:
            try:
                testfile = tempfile.TemporaryFile(dir=env_dir)
                testfile.close()
                return os.path.join(env_dir, name)
            except OSError:
                pass
        return None

    @property
    def access_cl_option(self):
        if self.config_use_default_conda_env_path:
            return "-n"
        else:
            return "-p"

    def find(self):
        return self._get_conda_env_path(self.conda_env_name)

    def create(self):
        if not self.environment_file:
            command = [self.config_command, "create", "-y"]
            if self.config_conda_forge:
                command += ["-c", "conda-forge", "--no-channel-priority"]
            command += [
                f"python={self.python_version}",
                "pip",
            ]
        elif self.config_command == "micromamba":
            command = ["micromamba", "create", "-y",
                       "--file", self.environment_file]
        else:
            command = [self.config_command, "env", "create", "--file", self.environment_file]
        if self.config_prefix is not None:
            command += ["--prefix", self.config_prefix]
        else:
            command += ["-n", self.conda_env_name]

        if self.verbosity > 0:  # no cov
            self.platform.check_command(command)
        else:
            self.platform.check_command_output(command)
        self.apply_env_vars()

    def remove(self):
        command = [self.config_command, "env", "remove", "-y"]

        if self.config_prefix is not None:
            command += ["--prefix", self.config_prefix]
        else:
            command += ["-n", self.conda_env_name]

        self.platform.check_command_output(command)

    def exists(self):
        env_path = self._get_conda_env_path(self.conda_env_name)
        if env_path is not None:
            return Path(self._get_conda_env_path(self.conda_env_name)).exists()
        return False

    def construct_conda_run_command(self, command):
        head = [self.config_command, "run"]
        if self.config_command in ["conda", "mamba"]:
            head.append("--no-capture-output")

        if self.config_prefix is not None:
            head += ["--prefix", self.config_prefix]
        else:
            head += ["-n", self.conda_env_name]

        return [*head, *command]

    def construct_pip_install_command(self, *args, **kwargs):
        return self.construct_conda_run_command(super().construct_pip_install_command(*args, **kwargs))

    def install_project(self):
        self.apply_env_vars()
        with self:
            self.platform.check_command(self.construct_pip_install_command(
                [self.apply_features(self.project_path)]))

    def install_project_dev_mode(self):
        self.apply_env_vars()
        with self:
            self.platform.check_command(
                self.construct_pip_install_command(
                    ["--editable", self.apply_features(self.project_path)])
            )

    def dependencies_in_sync(self):
        if not self.dependencies:
            return True
        self.apply_env_vars()
        with self:
            process = self.platform.run_command(
                " ".join(["hatchling", "dep", "synced", "-p",
                         "python", *self.dependencies]),
                capture_output=True,
            )
            return not process.returncode

    def sync_dependencies(self):
        self.apply_env_vars()
        with self:
            self.platform.check_command(
                self.construct_pip_install_command(self.dependencies))

    @contextmanager
    def command_context(self):
        with self:
            yield

    def run_shell_command(self, command):
        self.apply_env_vars()
        return self.platform.run_command(
            " ".join(
                self.construct_conda_run_command(
                    [
                        command,
                    ]
                )
            )
        )

    def enter_shell(self, name, path, args):  # no cov
        cmdl = f"{self.config_command} activate {self.conda_env_name}"
        shell_executor = getattr(self.shells, f"enter_{name}", None)
        if shell_executor is None:
            raise NotImplementedError(
                f"entering {name} shell in not supported yet")
        else:
            self.apply_env_vars()
            shell_executor(path, args, cmdl)

    def apply_env_vars(self):
        if self.config_command == "micromamba":
            for env_var, value in dict(self.env_vars).items():
                os.environ[env_var] = value
        else:
            env_vars = []
            for env_var, value in dict(self.env_vars).items():
                value_fixed = value
                if sys.platform == "win32":
                    value_fixed = value_fixed.replace("%", "%%%%%%%%")
                env_vars.append(f"{env_var}={value_fixed}")
            self.platform.check_command(
                ["conda", "env", "config", "vars", "set",
                    self.access_cl_option, self.conda_env_name, "--"] + env_vars
            )
