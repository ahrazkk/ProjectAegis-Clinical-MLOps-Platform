# DDI Model Tests

Comprehensive test suite for the Drug-Drug Interaction prediction model.

## Running Tests

### Run All Tests
```powershell
# Using pytest directly
pytest tests/test_tokenization.py -v

# Using test runner
python tests/run_tests.py
```

### Run Specific Test Class
```powershell
pytest tests/test_tokenization.py::TestDDITokenizerInitialization -v
```

### Run Specific Test
```powershell
pytest tests/test_tokenization.py::TestMarkDrugsInText::test_mark_drugs_basic -v
```

### Run with Coverage
```powershell
pytest tests/test_tokenization.py --cov=src.model.tokenization --cov-report=html
```

## Test Structure

### Test Classes

1. **TestDDITokenizerInitialization** - Tests tokenizer initialization
   - Default parameters
   - Custom max_length
   - Special token setup
   - NER labels
   - Vocabulary size

2. **TestMarkDrugsInText** - Tests drug marking in text
   - Basic marking
   - Reversed order
   - Adjacent drugs
   - With punctuation
   - Text structure preservation

3. **TestCreateDrugMask** - Tests drug mask creation
   - Basic mask
   - No markers
   - Only start marker
   - Adjacent markers
   - Tensor types

4. **TestCreateNerLabels** - Tests NER label generation
   - Basic labels
   - No drugs
   - Multiple drugs
   - Marker exclusion
   - Tensor types

5. **TestTokenize** - Tests tokenization
   - Output structure
   - Output types
   - Shape matching
   - Max length
   - Both drugs
   - Return tensors

6. **TestBatchTokenize** - Tests batch tokenization
   - Single sample
   - Multiple samples
   - Stacking
   - Consistent shapes
   - Empty list

7. **TestDecode** - Tests decoding
   - Basic decode
   - Special tokens
   - Drug markers

8. **TestResizeModelEmbeddings** - Tests embedding resizing
   - Correct method call
   - Correct size

9. **TestIntegration** - Integration tests
   - Mark and tokenize workflow
   - Batch workflow

## Test Coverage

Current test coverage for `tokenization.py`:
- Initialization: ✓ 100%
- Mark drugs in text: ✓ 100%
- Create drug mask: ✓ 100%
- Create NER labels: ✓ 100%
- Tokenize: ✓ 100%
- Batch tokenize: ✓ 100%
- Decode: ✓ 100%
- Resize embeddings: ✓ 100%

## Test Fixtures

### mock_tokenizer
Mock `AutoTokenizer` to avoid downloading large models during testing.
- Provides token ID mappings
- Mocks tokenization behavior
- Mocks decode functionality

### ddi_tokenizer
Fully initialized `DDITokenizer` with mocked dependencies.
- Ready to use in tests
- No model downloads required
- Fast test execution

## Writing New Tests

### Example Test
```python
def test_your_feature(ddi_tokenizer):
    """Test description"""
    # Arrange
    input_data = "test input"

    # Act
    result = ddi_tokenizer.some_method(input_data)

    # Assert
    assert result == expected_output
```

### Test Naming Convention
- `test_<method>_<scenario>` - e.g., `test_tokenize_with_both_drugs`
- Be descriptive and specific
- Focus on one behavior per test

## Continuous Integration

Tests are automatically run on:
- Every commit to `main` branch
- All pull requests
- Manual workflow dispatch

## Troubleshooting

### Import Errors
```powershell
# Ensure src is in PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"
pytest tests/
```

### Slow Tests
Tests use mocked tokenizers to avoid downloading models. If tests are slow:
- Check network connectivity
- Verify mocks are properly applied
- Clear pytest cache: `pytest --cache-clear`

### Test Failures
1. Check test output for specific failure
2. Run single test with `-v` flag
3. Use `--tb=long` for full traceback
4. Check that code changes didn't break assumptions

## Future Tests

Planned additions:
- [ ] Performance benchmarks
- [ ] Memory usage tests
- [ ] Edge case stress tests
- [ ] Property-based tests with Hypothesis
- [ ] Integration tests with real models (optional)
