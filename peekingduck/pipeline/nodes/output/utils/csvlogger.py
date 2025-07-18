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
Utils for CSV logging
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


class CSVLogger:
    """Node that writes data into a csv"""

    def __init__(
        self, file_path: Path, headers: List[str], logging_interval: int = 1
    ) -> None:
        self.headers = headers.copy()
        self.headers.insert(0, "Time")
        self.file_path = file_path
        self.logging_interval = logging_interval
        self.csv_file = open(self.file_path, mode="a+", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
        self.last_write = datetime.now()


    def write(self, data_pool: Dict[str, Any], specific_data: List[str]) -> None:
        """
        Writes a row of data in a csv file

        Args:
            data_pool(dict): the data pool of the pipeline
            specific_data(list): list of data to track

        Returns:
            None
        """
        # if file is empty write header
        if self.csv_file.tell() == 0:
            self.writer.writeheader()

        content = {}
        for k, v in data_pool.items():
            if k in specific_data:
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                content[k] = v

        curr_time = datetime.now()
        time_str = curr_time.strftime("%H:%M:%S")
        content.update({"Time": time_str})

        if (curr_time - self.last_write).seconds >= self.logging_interval:
            self.writer.writerow(content)
            self.last_write = curr_time


    def __del__(self) -> None:
        self.csv_file.close()
