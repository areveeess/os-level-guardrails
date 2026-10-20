"""
Shell Splitter — see every command the shell would run
=======================================================
The policy engine reads a *string*; /bin/sh interprets that string. If the two
disagree about what the string means, the policy can be bypassed (a "parse
differential"). The original engine looked only at the first word, so

    ls; curl http://evil.example

was judged to be `ls` (allowed) while the shell went on to run `curl`.

This module closes the gap by doing what the shell does first: it splits the
string into *simple commands* — separated by ; && || | |& & and newlines, and
by grouping parentheses — honouring quoting and escapes, and removing leading
NAME=value assignments and shell keywords. The engine then checks every
simple command, and the first blocked one blocks the whole string.

Rule of thumb: anything this module cannot parse with confidence raises
ShellParseError, and the engine refuses it (fail closed). Over-splitting only
ever makes the verdict stricter, never looser.
"""

import re
from dataclasses import dataclass, field
from typing import List


class ShellParseError(ValueError):
    """The input cannot be parsed safely; the caller must refuse it."""


@dataclass
class SimpleCommand:
    """One command the shell would run."""
    text: str                                            # source text of this command
    argv: List[str] = field(default_factory=list)        # command name + arguments
    writes: List[str] = field(default_factory=list)      # output-redirection targets
    reads: List[str] = field(default_factory=list)       # input-redirection targets


_ASSIGNMENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*=')
_KEYWORDS = {'if', 'then', 'else', 'elif', 'fi', 'do', 'done',
             'while', 'until', '!', '{', '}'}
_FD_TARGET = re.compile(r'\d+-?|-')                      # `2>&1`, `>&-`: descriptor, not a file


@dataclass
class _Word:
    text: str


class _Parser:
    """Single-pass, quote-aware scanner over one command string."""

    def __init__(self, source: str):
        self.s = source
        self.results: List[SimpleCommand] = []
        self._reset_command(0)
        self.paren_depth = 0

    # ── per-command state ──

    def _reset_command(self, start: int):
        self.words: List[_Word] = []
        self.chars: List[str] = []
        self.in_word = False
        self.writes: List[str] = []
        self.reads: List[str] = []
        self.pending = None          # kind of redirection waiting for its target
        self.cmd_start = start

    def _add_text(self, text: str):
        self.chars.append(text)
        self.in_word = True

    def _end_word(self):
        if not self.in_word:
            return
        word = _Word(''.join(self.chars))
        self.chars = []
        self.in_word = False
        if self.pending is None:
            self.words.append(word)
            return
        kind, self.pending = self.pending, None
        if kind == 'ignore':
            return
        if kind in ('dupw', 'dupr') and _FD_TARGET.fullmatch(word.text):
            return
        (self.writes if kind in ('w', 'dupw') else self.reads).append(word.text)

    def _flush(self, end: int, next_start: int):
        """A separator was reached: close the current simple command."""
        self._end_word()
        if self.pending is not None:
            raise ShellParseError("redirection without a target")
        words = self.words
        while words and (_ASSIGNMENT.match(words[0].text) or words[0].text in _KEYWORDS):
            words = words[1:]
        text = self.s[self.cmd_start:end].strip()
        if words or self.writes or self.reads:
            self.results.append(SimpleCommand(
                text=text, argv=[w.text for w in words],
                writes=self.writes, reads=self.reads))
        self._reset_command(next_start)

    # ── redirections ──

    def _redirect(self, i: int) -> int:
        """Handle a < or > at index i; return the index after the operator."""
        s = self.s
        # `2>file`: a purely numeric word glued to the operator is a descriptor
        if self.in_word and ''.join(self.chars).isdigit():
            self.chars, self.in_word = [], False
        else:
            self._end_word()
        if self.pending is not None:
            raise ShellParseError("redirection without a target")
        if s[i] == '>':
            if s.startswith('>>', i):
                self.pending, j = 'w', i + 2
            elif s.startswith('>|', i):
                self.pending, j = 'w', i + 2
            elif s.startswith('>&', i):
                self.pending, j = 'dupw', i + 2
            else:
                self.pending, j = 'w', i + 1
        else:
            if s.startswith('<<<', i):
                self.pending, j = 'ignore', i + 3          # here-string: data, not a file
            elif s.startswith('<<', i):
                raise ShellParseError("here-document is not supported")
            elif s.startswith('<&', i):
                self.pending, j = 'dupr', i + 2
            elif s.startswith('<>', i):
                self.pending, j = 'w', i + 2
            else:
                self.pending, j = 'r', i + 1
        return j

    # ── main scan ──

    def run(self) -> List[SimpleCommand]:
        s, n, i = self.s, len(self.s), 0
        while i < n:
            c = s[i]
            if c == '\\':
                if i + 1 >= n:
                    raise ShellParseError("trailing backslash")
                if s[i + 1] != '\n':                      # backslash-newline is a continuation
                    self._add_text(s[i + 1])
                i += 2
            elif c == "'":
                j = s.find("'", i + 1)
                if j < 0:
                    raise ShellParseError("unbalanced single quote")
                self._add_text(s[i + 1:j])
                i = j + 1
            elif c == '"':
                i = self._double_quoted(i)
            elif c in ' \t\r':
                self._end_word()
                i += 1
            elif c in ';\n':
                self._flush(i, i + 1)
                i += 1
            elif c == '&':
                if s.startswith('&>', i):                 # &> and &>> redirect both streams
                    self._end_word()
                    self.pending = 'w'
                    i += 3 if s.startswith('&>>', i) else 2
                else:
                    step = 2 if s.startswith('&&', i) else 1
                    self._flush(i, i + step)
                    i += step
            elif c == '|':
                step = 2 if s[i + 1:i + 2] in ('|', '&') else 1
                self._flush(i, i + step)
                i += step
            elif c == '(':
                self.paren_depth += 1
                self._flush(i, i + 1)
                i += 1
            elif c == ')':
                if self.paren_depth == 0:
                    raise ShellParseError("unbalanced parenthesis")
                self.paren_depth -= 1
                self._flush(i, i + 1)
                i += 1
            elif c in '<>':
                i = self._redirect(i)
            else:
                self._add_text(c)
                i += 1
        if self.paren_depth:
            raise ShellParseError("unbalanced parenthesis")
        self._flush(n, n)
        return self.results

    def _double_quoted(self, i: int) -> int:
        """Consume a double-quoted span starting at index i; return the index after it."""
        s, n = self.s, len(self.s)
        j, out = i + 1, []
        while True:
            if j >= n:
                raise ShellParseError("unbalanced double quote")
            d = s[j]
            if d == '"':
                break
            if d == '\\' and j + 1 < n:
                e = s[j + 1]
                if e in '$`"\\':
                    out.append(e)
                elif e != '\n':
                    out.append('\\' + e)
                j += 2
                continue
            out.append(d)
            j += 1
        self._add_text(''.join(out))
        return j + 1


def parse_shell(command: str) -> List[SimpleCommand]:
    """Return every simple command the shell would run for `command`.

    Raises ShellParseError for anything that cannot be parsed safely.
    """
    return _Parser(command).run()
