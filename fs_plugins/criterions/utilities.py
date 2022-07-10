##########################################################################
# Copyright (C) 2022 COAI @ Tsinghua University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

def parse_anneal_argument(anneal_str):
    def parse_value_pos(value_str):
        if "@" in value_str:
            value, pos = value_str.split("@")
        else:
            value = value_str
            pos = "0"
        return float(value), float(pos.replace("k", "000"))

    res = []
    for value_str in anneal_str.split(":"):
        res.append(parse_value_pos(value_str))
    return res

def get_anneal_value(anneal_params, update_num):
    last_value, last_pos = anneal_params[0][0], 0
    for value, pos in anneal_params:
        if update_num < pos:
            return last_value + (value - last_value) * (update_num - last_pos) / (pos - last_pos + 1)
        last_value, last_pos = value, pos
    return anneal_params[-1][0]
