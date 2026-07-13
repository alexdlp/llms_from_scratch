"""
Terminal coloring utilities for log messages.
Provides ANSI-wrapped color functions for readable, standardized output.
"""

# ANSI escape sequences
_RESET = "\033[0m"
_BOLD = "\033[1m"

_COLORS = {
    "black":   "\033[30m",
    "red":     "\033[31m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "cyan":    "\033[36m",
    "white":   "\033[37m",
    "orange": "\033[38;5;208m",
}

def color(text: str, color_name: str) -> str:
    """
    Generic color wrapper using ANSI escape codes.

    Args:
        text (str): Text to colorize.
        color_name (str): One of: black, red, green, yellow, blue,
                          magenta, cyan, white.

    Returns:
        str: Colored string with ANSI codes.
    """
    if color_name not in _COLORS:
        return text
    return f"{_COLORS[color_name]}{text}{_RESET}"

def green(text: str) -> str:
    """Shortcut for green text."""
    return f"{_COLORS['green']}{text}{_RESET}"

def red(text: str) -> str:
    """Shortcut for red text."""
    return f"{_COLORS['red']}{text}{_RESET}"

def yellow(text: str) -> str:
    """Shortcut for yellow text."""
    return f"{_COLORS['yellow']}{text}{_RESET}"

def magenta(text: str) -> str:
    """Shortcut for magenta text."""
    return f"{_COLORS['magenta']}{text}{_RESET}"

def orange(text: str) -> str:
    """Shortcut for magenta text."""
    return f"{_COLORS['orange']}{text}{_RESET}"

def bold(text: str) -> str:
    """Shortcut for bold text."""
    return f"{_BOLD}{text}{_RESET}"

def bold_color(text: str, color_name: str) -> str:
    """
    Combines bold style with a selected color.

    Args:
        text (str): Text to format.
        color_name (str): Color name defined in _COLORS.

    Returns:
        str: ANSI-formatted bold + color text.
    """
    if color_name not in _COLORS:
        return text
    return f"{_BOLD}{_COLORS[color_name]}{text}{_RESET}"


def bold_green(text: str) -> str:
    return f"{_BOLD}{_COLORS['green']}{text}{_RESET}"

def bold_red(text: str) -> str:
    return f"{_BOLD}{_COLORS['red']}{text}{_RESET}"

def bold_yellow(text: str) -> str:
    return f"{_BOLD}{_COLORS['yellow']}{text}{_RESET}"