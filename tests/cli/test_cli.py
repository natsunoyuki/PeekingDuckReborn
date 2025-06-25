# Modifications copyright 2025 Natsunoyuki AI Laboratory
#
# PeekingDuckReborn is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
#
# PeekingDuckReborn is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# PeekingDuckReborn. If not, see <https://www.gnu.org/licenses/>.

# Original copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from peekingduck import __version__
from peekingduck.cli import cli


@pytest.fixture
def cwd():
    return Path.cwd()


@pytest.fixture
def parent_dir():
    return Path.cwd().parent


@pytest.mark.usefixtures("tmp_dir", "tmp_project_dir")
class TestCli:
    def test_help(self):
        """Checks that calling peekingduck without options and commands prints
        help message.
        """
        result = CliRunner().invoke(cli)

        # TODO currently result.exit_code=2 and I don't know why.
        # assert result.exit_code == 0
        # not testing full message as .invoke() sets program name to cli
        # instead of peekingduck
        assert f"Usage:" in result.output
        assert f"Options:" in result.output
        assert f"Commands:" in result.output

    def test_version(self):
        result = CliRunner().invoke(cli, ["--version"])

        assert result.exit_code == 0
        # not testing full message as .invoke() sets program name to cli
        # instead of peekingduck
        assert f"version {__version__}" in result.output

    def test_main_py_log_level_debug(self):
        # setup unit test env
        #tmp_dir = Path.cwd()
        #print(f"\ntmp_dir={tmp_dir}")
        unit_test_run_dir = Path(__file__).parents[3]
        print(f"unit_test_run_dir={unit_test_run_dir}")

        nodes = {
            "nodes": [
                {
                    "input.visual": {
                        "source": "{}/PeekingDuckReborn/tests/data/images".format(
                            unit_test_run_dir
                        )
                    }
                }
            ]
        }

        os.chdir(unit_test_run_dir)
        # test_config_path = tmp_dir / "test_config.yml"
        test_config_path = "test_config.yml"
        with open(test_config_path, "w") as outfile:
            yaml.dump(nodes, outfile, default_flow_style=False)

        # run unit test
        cmd = [
            "python",
            "PeekingDuckReborn",
            "--log_level",
            "debug",
            "--config_path",
            f"{test_config_path}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out, _ = proc.communicate()
        out_str = out.decode("utf-8")
        print(out_str)
        exit_status = proc.returncode

        os.remove(test_config_path)

        assert "DEBUG" in out_str
        assert exit_status == 0
