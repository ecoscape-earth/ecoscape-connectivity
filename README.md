# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/ecoscape-earth/ecoscape-connectivity/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                    |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| ecoscape\_connectivity/\_\_init\_\_.py  |        4 |        0 |        0 |        0 |    100% |           |
| ecoscape\_connectivity/distributions.py |       27 |        3 |        2 |        0 |     90% |     13-16 |
| ecoscape\_connectivity/repopulation.py  |      185 |       10 |       74 |       10 |     91% |148-151, 449-\>452, 532-534, 555-\>563, 560-\>563, 608-\>618, 613-\>618, 619-621, 623, 625 |
| ecoscape\_connectivity/util.py          |       38 |        8 |        6 |        0 |     77% |25-30, 58-60 |
| **TOTAL**                               |  **254** |   **21** |   **82** |   **10** | **89%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/ecoscape-earth/ecoscape-connectivity/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/ecoscape-earth/ecoscape-connectivity/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ecoscape-earth/ecoscape-connectivity/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/ecoscape-earth/ecoscape-connectivity/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fecoscape-earth%2Fecoscape-connectivity%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/ecoscape-earth/ecoscape-connectivity/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.