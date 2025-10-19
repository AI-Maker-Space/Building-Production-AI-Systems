# Run specific test files

```bash
python run_tests.py --file test_rag_graph.py
python run_tests.py --file test_app.py
```

# Run by test type

```bash
python run_tests.py --type fast
python run_tests.py --type unit
python run_tests.py --type integration
```

# Additional options
```bash
python run_tests.py --file test_rag_graph.py --coverage
python run_tests.py --type fast --quiet
python run_tests.py --help
```