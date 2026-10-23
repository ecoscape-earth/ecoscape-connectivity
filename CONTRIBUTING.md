
## Setting up

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
   and the GDAL library (for example, `brew install gdal` on macOS). The
   GDAL version must match `constraint-dependencies` in `pyproject.toml`;
   check yours with `gdal-config --version`.
2. From the repository folder, install the exact dependency versions
   from `uv.lock` into `.venv`:

       make install

   uv downloads the Python version in `.python-version` if you don't
   have it.
3. Check that everything works:

       make test

### VS Code

Open the Command Palette (⇧⌘P / Ctrl+Shift+P), run
**Python: Select Interpreter**, and choose `.venv/bin/python`. Use the
same environment as the kernel for notebooks. VS Code then runs and
debugs tests with the project's dependencies.

## Managing dependencies

Dependencies are listed in `pyproject.toml`, and `uv.lock` pins every
package, including indirect ones, to an exact version. This means
everyone gets the same versions, and results can be reproduced.

- Run `make install` after pulling. It installs exactly the versions in
  `uv.lock`, and never upgrades anything on its own.
- Add or remove a dependency with `uv add <name>` or `uv remove <name>`
  (development tools: `uv add --dev <name>`), rather than editing
  `pyproject.toml` by hand. uv updates `uv.lock` too.
- To upgrade a package on purpose, run
  `uv lock --upgrade-package <name>`, then `make test`. Upgrade in a pull
  request of its own, so any change in results can be traced to it.
- Always commit `uv.lock` in the same commit as the `pyproject.toml`
  change. Automatic checks install with `--locked`, which fails if the
  two don't match.
- Before committing, `uv lock --check` confirms `uv.lock` is up to date.
- Don't use `pip install` in `.venv`. uv doesn't know about packages
  installed that way, and the next `make install` removes them.

## Tasks

To build the documentation:

    make docs

This regenerates `Documentation.md` from the docstrings and `README.md`
from `README-prototype.md` and `Documentation.md`. Don't edit `README.md`
directly.

To build the package:

    uv build

To publish it to PyPI:

    uv publish

## Making a change

Every change starts from a GitHub issue and goes through a pull request.

1. Create a branch from `main`, named `<issue-number>-<short-description>`.
   For example, for issue #9 "Fix the existing tests so they run again":

       git checkout main
       git pull
       git checkout -b 9-fix-tests

2. Make your changes and commit them to your branch.

3. Push your branch to GitHub:

       git push -u origin 9-fix-tests

4. Create a pull request from your branch into `main`, either on GitHub
   or with:

       gh pr create

5. Write the pull request description. GitHub fills in a template with
   these sections when you create the pull request
   (see `.github/pull_request_template.md`). Mention the issue so GitHub links
   the two; writing `Closes #9` closes the issue automatically when the
   pull request is merged. The description should include:

   - **Motivation**: why the change is needed.
   - **Summary of changes**: what you changed.
   - **What reviewers should look for**: anything that needs careful
     review, such as tricky logic or changes to scientific results.
   - **How to test**: the steps or commands a reviewer can run to check
     the change works.

   For example:

       Closes #9

       ## Motivation
       The existing tests fail because function inputs have changed.

       ## Summary of changes
       - Updated tests to match the current function signatures.
       - Added instructions for running tests to the README.

       ## What reviewers should look for
       Check that the updated test inputs still match how the
       functions are used in practice.

       ## How to test
       Run the tests and confirm they all pass.
