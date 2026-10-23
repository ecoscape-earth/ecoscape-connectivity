Do not run Python code on the command line (ie `python3 -c "...""`` or `uv run python -c "..."`).  When writing code, append it as a function to claude.py. Include a comment about the purpose of the code.

When writing Python code

  - Always use PEP-8 style.
  - Format comments according to PEP 257 conventions.
  - Do not use inline imports.
  - Always run `make lint` after editing code.
  - Add type annotations to all function parameters and return values.
  - Annotate variables only when the assignment doesn't make the type
    obvious. Omit annotations for literals (`SCALE_FONT_SIZE = 6.0`) and
    constructor calls (`figure = Figure(...)`). Keep them for empty
    containers (`texts: list[Text] = []`) and for values returned by
    functions or methods.
  - Don't define a constant for a value used only once; write the value
    inline, with a comment explaining it if the meaning isn't obvious.
    Define a constant only when several places share the value.

Dependencies

  - The project uses uv. Check for dependencies in pyproject.toml (`dependencies` for the package, the `dev` group under `[dependency-groups]` for development tools) and uv.lock (exact installed versions); do NOT run pip or attempt to import to verify package install.
  - Suggest new dependencies when they make code simpler or easier to read. Claude can recommend new dependencies, with the `uv add <name>` or `uv add --dev <name>` command for the user to run, but only the user can install them. claude is NEVER allowed to install packages directly: do not run `uv add`, `uv remove`, `uv sync`, `uv pip`, `make install` or `uv lock --upgrade`. Read-only checks such as `uv lock --check` are ok.
  - `uv run` (and so `make test` and `make docs`) creates or updates `.venv` when it is missing or out of date with uv.lock, which installs packages. If `.venv` is missing or pyproject.toml or uv.lock has changed, ask the user to run `make install` first.

Prefer makefile targets whenever possible. For example, run `make lint` instead of `black main.py`

Claude is *not* allowed to run any git commands that write, change, or destroy data. For example, do not run `git commit`; it is ok to run `git status` or other read-only commands. It is ok to make changes to git-controlled files. When proposing commits, write a commit message to a markdown file for review. Ask about any other git-related write operations.