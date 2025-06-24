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

"""
To test "import peekingduck" as a module
"""

import pytest
import subprocess
import textwrap
from pathlib import Path

# This version is read by CI/CD script and used for post-merge tests
TEST_VERSION = "0.0.0dev"


@pytest.mark.module
@pytest.mark.usefixtures("tmp_dir")
class TestModuleImport:
    """
    Technotes:
    1. The code scripts below do 'import peekingduck ...'
       This will import the installed version of Peeking Duck in the test environment.
    2. The CI/CD post-merge test script install this development version in the
       test environment using `pip install . --no-dependencies`, so the test code
       below will be run on this development version instead of the released version.
    """

    @staticmethod
    def exec_code(code: str) -> str:
        """Helper to execute given code string

        Args:
            code (str): python program to run

        Returns:
            str: output from running given code
        """
        temp_dir = Path.cwd()
        with open("tmp.py", "w") as f:
            f.writelines(code)
        cmd = ["python", "tmp.py"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=temp_dir)
        out, err = proc.communicate()
        assert err is None  # we don't want any errors
        out_str = out.decode("utf-8")
        return out_str

    def test_version(self):
        """
        Check we get a valid version number
        """
        code = textwrap.dedent(
            """
            from peekingduck import __version__ as ver

            print(ver)
            """
        )
        output = TestModuleImport.exec_code(code)
        res = TEST_VERSION == output
        assert res is not None

    def test_pkd_imports(self):
        """
        Check can import without errors
        """
        code = textwrap.dedent(
            """
            from peekingduck import cli
            from peekingduck.config_loader import ConfigLoader
            from peekingduck.declarative_loader import DeclarativeLoader
            from peekingduck.runner import Runner
            from peekingduck.pipeline.pipeline import Pipeline
            from peekingduck.pipeline.nodes.abstract_node import AbstractNode

            print("good")
            """
        )
        output = TestModuleImport.exec_code(code)
        res = "good" == output
        assert res is not None

    def test_pkd_import_types(self):
        """
        Check imports are of the right (class) types
        """
        code = textwrap.dedent(
            """
            from peekingduck import cli
            from peekingduck.config_loader import ConfigLoader
            from peekingduck.declarative_loader import DeclarativeLoader
            from peekingduck.runner import Runner
            from peekingduck.pipeline.pipeline import Pipeline
            from peekingduck.pipeline.nodes.abstract_node import AbstractNode
            import inspect

            the_types = []
            the_types.append(inspect.isclass(cli))
            the_types.append(inspect.isclass(ConfigLoader))
            the_types.append(inspect.isclass(DeclarativeLoader))
            the_types.append(inspect.isclass(Runner))
            the_types.append(inspect.isclass(Pipeline))
            the_types.append(inspect.isclass(AbstractNode))
            the_types_str = "".join([str(x) for x in the_types])
            print(the_types_str)
            """
        )
        output = TestModuleImport.exec_code(code)
        print(output)
        res = "FalseTrueTrueTrueTrueTrue" == output
        assert res is not None
