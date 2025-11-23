# Contributing to Vietnamese Banking Stock Predictor

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/banking-stock-predictor.git
   cd banking-stock-predictor
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Run setup**:
   ```bash
   python setup.py
   ```

## Code Style

We follow PEP 8 guidelines with some modifications:

- **Line length**: 120 characters
- **Formatter**: Black
- **Import sorting**: isort
- **Linter**: flake8

### Format your code

```bash
make format
```

### Check code quality

```bash
make lint
```

## Testing

### Run tests

```bash
make test
```

### Write tests

- Place tests in the `tests/` directory
- Use pytest fixtures for common setup
- Aim for >80% code coverage
- Test both success and failure cases

Example test structure:
```python
def test_feature_engineering():
    """Test feature engineering pipeline."""
    # Arrange
    engineer = FeatureEngineer()
    
    # Act
    result = engineer.process_ticker('VCB')
    
    # Assert
    assert result is not None
    assert 'features' in result
```

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example:
```
feat: add support for quarterly predictions

- Implement quarterly feature aggregation
- Add new model architecture for long-term predictions
- Update config with quarterly parameters
```

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   make test
   make lint
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**:
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all tests pass

## Code Review Guidelines

### For Contributors

- Respond to feedback promptly
- Be open to suggestions
- Keep PRs focused and reasonably sized

### For Reviewers

- Be constructive and respectful
- Focus on code quality and maintainability
- Test the changes locally when possible

## Areas for Contribution

### High Priority

- [ ] Additional model architectures (LSTM, GRU, etc.)
- [ ] More comprehensive tests
- [ ] Performance optimization
- [ ] Documentation improvements

### Medium Priority

- [ ] Additional technical indicators
- [ ] Enhanced visualization features
- [ ] API endpoints for predictions
- [ ] Docker containerization

### Low Priority

- [ ] Additional data sources
- [ ] Mobile-friendly UI
- [ ] Real-time prediction updates
- [ ] Multi-language support

## Documentation

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Keep CHANGELOG.md updated

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
