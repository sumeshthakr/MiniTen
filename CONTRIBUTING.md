# Contributing to MiniTen

Thank you for your interest in contributing to MiniTen! This document provides guidelines and instructions for contributing.

## 🎯 Project Goals

MiniTen aims to be:
1. **Educational**: Clear, well-documented code showing how deep learning works
2. **Performant**: Highly optimized for edge devices
3. **Minimal**: Small footprint with minimal dependencies
4. **Robust**: Well-tested and reliable
5. **Collaborative**: Easy for others to understand and contribute to

## 🚀 Getting Started

### Setting Up Development Environment

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/MiniTen.git
cd MiniTen
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

4. **Build Cython extensions**
```bash
python setup.py build_ext --inplace
```

5. **Run tests**
```bash
python -m pytest tests/ -v
```

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Write docstrings for all public functions and classes
- Add type hints where appropriate
- Keep functions focused and modular

### Documentation

- Document all public APIs with clear docstrings
- Include examples in docstrings
- Update README.md if adding major features
- Add comments for complex algorithms or optimizations
- Explain the "why" not just the "what"

### Testing

- Write tests for all new features
- Ensure existing tests pass
- Aim for high test coverage
- Include edge cases in tests
- Test on multiple platforms if possible

### Optimization

- Profile code before optimizing
- Document optimization rationale
- Benchmark performance improvements
- Consider edge device constraints (memory, power)
- Use Cython for performance-critical code

## 🔧 Types of Contributions

### 1. Bug Fixes
- Check existing issues before starting
- Include a test that reproduces the bug
- Explain the fix in the PR description

### 2. New Features
- Open an issue to discuss the feature first
- Follow existing code patterns
- Add comprehensive tests
- Update documentation
- Ensure backward compatibility

### 3. Optimizations
- Provide benchmarks showing improvement
- Explain the optimization technique
- Ensure correctness is maintained
- Consider trade-offs (speed vs memory)

### 4. Documentation
- Fix typos and clarify explanations
- Add examples and tutorials
- Improve docstrings
- Create guides for common use cases

### 5. Testing
- Increase test coverage
- Add edge case tests
- Add performance benchmarks
- Test on edge devices

## 📋 Pull Request Process

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write clear, focused commits
   - Follow code style guidelines
   - Add tests for new functionality

3. **Test your changes**
```bash
python setup.py build_ext --inplace
python -m pytest tests/ -v
python test_vector_operations.py
python test_backprop.py
```

4. **Update documentation**
   - Update docstrings
   - Update README if needed
   - Add examples if appropriate

5. **Commit your changes**
```bash
git add .
git commit -m "Add feature: clear description"
```

6. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

7. **Open a Pull Request**
   - Provide a clear title and description
   - Reference related issues
   - Explain what changed and why
   - Include test results
   - Add benchmarks if relevant

### PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commits are clear and focused
- [ ] No unnecessary files committed
- [ ] Branch is up to date with main

## 🎨 Code Areas

### High Priority
- [ ] Complete Tensor implementation
- [ ] Automatic differentiation engine
- [ ] GPU backend implementations
- [ ] Core neural network layers
- [ ] Loss functions
- [ ] Optimizers

### Medium Priority
- [ ] RNN/LSTM/GRU implementations
- [ ] CNN optimizations
- [ ] Data loading utilities
- [ ] Image processing
- [ ] Audio processing

### Lower Priority (but welcome!)
- [ ] Additional examples
- [ ] Tutorial notebooks
- [ ] Performance benchmarks
- [ ] Documentation improvements
- [ ] Edge device testing

## 🐛 Reporting Bugs

### Before Submitting
- Check if the bug has already been reported
- Try to reproduce with the latest version
- Collect relevant information

### Bug Report Template
```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce the behavior:
1. Code to run
2. Expected output
3. Actual output

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9]
- MiniTen version: [e.g., 0.1.0]
- Hardware: [e.g., Raspberry Pi 4, Jetson Nano]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Before Submitting
- Check if feature has been requested
- Consider if it fits project goals
- Think about edge device constraints

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed?
What problem does it solve?

**Implementation Ideas**
Suggestions for how to implement

**Alternatives**
Alternative solutions you've considered

**Edge Device Considerations**
How will this work on resource-constrained devices?
```

## 🏗️ Architecture Guidelines

### Cython Usage
- Use Cython for performance-critical operations
- Add type declarations for optimization
- Use memoryviews for arrays
- Profile to verify improvement

### Memory Management
- Be mindful of memory allocations
- Reuse buffers where possible
- Consider in-place operations
- Test on memory-constrained devices

### GPU Backend
- Support multiple backends (CUDA, OpenCL, Metal, Vulkan)
- Provide CPU fallback
- Make GPU operations optional
- Optimize for edge GPUs (Jetson, mobile)

## 📚 Resources

### Learning Resources
- [Cython Documentation](https://cython.readthedocs.io/)
- [Understanding Backpropagation](http://neuralnetworksanddeeplearning.com/)
- [Edge Computing Optimization](https://www.tensorflow.org/lite/performance/best_practices)

### Similar Projects
- PyTorch (design inspiration)
- TensorFlow Lite (edge optimization)
- Tinygrad (minimalism)

## 🤝 Community

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Pull Requests: Code contributions

### Code of Conduct
- Be respectful and inclusive
- Help others learn and grow
- Give constructive feedback
- Focus on the code, not the person
- Acknowledge contributions

## 📄 License

By contributing to MiniTen, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## ❓ Questions?

If you have questions about contributing:
1. Check existing documentation
2. Search closed issues
3. Open a discussion on GitHub
4. Ask in your pull request

Thank you for contributing to MiniTen! 🎉
