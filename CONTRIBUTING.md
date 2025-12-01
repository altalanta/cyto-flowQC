# Contributing to CytoFlow-QC

First off, thank you for considering contributing to CytoFlow-QC! It's people like you that make this such a great tool.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/cytoflow-qc.git
    cd cytoflow-qc
    ```
3.  **Set up the development environment.** We use Poetry for dependency management and pre-commit for code quality. The `setup-dev` command in the `Makefile` will get you set up.
    ```bash
    make setup-dev
    ```
    This will install all dependencies into a local `.venv` directory and set up the pre-commit hooks, which will automatically format and lint your code before you commit.

## Making Changes

1.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b your-feature-branch-name
    ```
2.  **Make your changes.** Write clean, readable code and include tests for any new functionality.
3.  **Ensure all pre-commit checks pass.** The hooks will run automatically when you commit. If they fail, you'll need to fix the issues and re-add the files to your commit. You can also run the checks manually at any time:
    ```bash
    make lint-fix
    ```
4.  **Run the full test suite** to ensure your changes haven't introduced any regressions:
    ```bash
    make test
    ```

## Submitting a Pull Request

1.  **Push your branch** to your fork on GitHub:
    ```bash
    git push origin your-feature-branch-name
    ```
2.  **Open a pull request** to the `main` branch of the original repository.
3.  **Provide a clear description** of your changes in the pull request. Explain the problem you're solving and the approach you've taken.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

Thank you for your contribution!


