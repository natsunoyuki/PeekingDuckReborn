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

"""Utility helper methods for PeekingDuck Viewer"""

from PIL import ImageTk, Image


def get_keyboard_char(char: str, keysym: str) -> str:
    """Get keyboard character

    Args:
        char (str): Tk keypress event character
        keysym (str): Tk keypress event key symbol

    Returns:
        str: keyboard character
    """
    res = char if char else keysym
    if keysym == "minus":
        res = "-"
    if keysym == "plus":
        res = "+"
    return res


def get_keyboard_modifier(state: int) -> str:
    """Get keyboard modifier keys: support ctrl, alt/option, shift

    Args:
        state (int): Tk keypress event key state

    Returns:
        str: detected modifier keys

    Technotes:
        Modifier = {
            0x1    : 'Shift'            ,
            0x2    : 'CapsLock'         ,
            0x4    : 'Control'          ,
            0x8    : 'Left-Alt'         ,   #Mod1 (meta / mac Left-Option)
            0x10   : 'NumLock'          ,   #Mod2? (alt?)
            0x20   : 'ScrollLock'       ,   #Mod3? (keypad?)
            0x40   : 'Mod4'             ,   #?
            0x80   : 'Right-Alt'        ,   #Mod5 (meta / mac Right-Option)
            0x100  : 'MouseButton1'     ,
            0x200  : 'MouseButton2'     ,
            0x400  : 'MouseButton3'     ,
            0x800  : 'Button4'          ,
            0x1000 : 'Button5'          ,
            0x20000: 'Alt'              ,   #not a typo
        }
    """
    ctrl = (state & 0x4) != 0
    alt = (state & 0x8) != 0 or (state & 0x80) != 0
    shift = (state & 0x1) != 0
    res = f"{'ctrl' if ctrl else ''}{'-alt' if alt else ''}{'-shift' if shift else ''}"
    return "" if len(res) == 0 else res[1:] if res[0] == "-" else res


def load_image(image_path: str, resize_pct: float = 1.0) -> ImageTk.PhotoImage:
    """Load and resize an image, 'coz plain vanilla Tkinter doesn't support JPG, PNG

    Args:
        resize_pct (float, optional): percentage to resize.
                                      Defaults to original size 1.0.

    Returns:
        ImageTk.PhotoImage: the loaded image
    """
    img = Image.open(image_path)
    if resize_pct != 1.0:
        width = int(resize_pct * img.size[0])
        height = int(resize_pct * img.size[1])
        resized_img = img.resize((width, height))
    else:
        resized_img = img
    the_img = ImageTk.PhotoImage(resized_img)
    return the_img
