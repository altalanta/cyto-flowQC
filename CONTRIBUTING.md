# Contributing to CytoFlow-QC

We welcome contributions from the community! Whether you're fixing a bug, improving documentation, or proposing a new feature, your help is valued. Please take a moment to review this document to ensure a smooth and effective contribution process.

## Getting Started

1.  **Fork the repository:** Start by forking the [main repository](https://github.com/cytoflow-qc/cytoflow-qc) on GitHub.
2.  **Clone your fork:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/cytoflow-qc.git
    cd cytoflow-qc
    ```
3.  **Set up the development environment:** We use Poetry for dependency management.
    ```bash
    make setup
    ```
    This will install all dependencies, including development tools, and set up pre-commit hooks.

## Development Workflow

1.  **Create a new branch:** Create a descriptive branch name for your changes.
    ```bash
    git checkout -b feature/my-new-feature
    # or
    git checkout -b fix/bug-fix-description
    ```
2.  **Make your changes:** Write your code, following the existing code style.
3.  **Ensure code quality:** Before committing, run the quality assurance checks:
    ```bash
    make lint   # Check for linting errors
    make format # Automatically format the code
    make test   # Run the test suite
    ```
4.  **Commit your changes:** Write a clear and concise commit message.
    ```bash
    git commit -m "feat: Add new feature"
    ```
5.  **Push to your fork:**
    ```bash
    git push origin feature/my-new-feature
    ```

## Submitting a Pull Request

1.  Open a pull request from your fork to the `main` branch of the original repository.
2.  Provide a clear title and description of your changes. If your PR addresses an existing issue, please reference it (e.g., `Closes #123`).
3.  The maintainers will review your pull request. Please be responsive to any feedback or requested changes.

## Visual Regression Testing

We use `pytest-mpl` for visual regression testing of our plots. This ensures that visual outputs remain consistent.

-   **Running the tests:**
    ```bash
    poetry run pytest tests/test_visuals.py
    ```
-   **Updating baseline images:** If you make intentional changes to a plot, the visual tests will fail. To update the baseline images, run:
    ```bash
    poetry run pytest tests/test_visuals.py --mpl-generate-path=tests/baseline
    ```
    Then, commit the new baseline images in the `tests/baseline` directory along with your code changes.

## Code Style

- We use `black` for code formatting and `ruff` for linting. The pre-commit hooks will help enforce this, and you can run `make format` to apply the correct style.
- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for naming conventions and docstrings.

Thank you for contributing!


