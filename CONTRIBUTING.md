
## Tasks

To build locally: 

    python3 -m build

To upload: 

    python3 -m twine upload dist/*

To build the documentation:

    make docs

This regenerates `Documentation.md` from the docstrings and `README.md`
from `README-prototype.md` and `Documentation.md`. Don't edit `README.md`
directly.

To install locally: 

    python3 -m pip install . 

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
