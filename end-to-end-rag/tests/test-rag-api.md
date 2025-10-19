# Run specific test files

```bash
cd tests
python run_tests.py --type fast 
python run_tests.py --file tests/test_rag_graph.py
python run_tests.py --file tests/test_app.py
```

# Run by test type

```bash
python run_tests.py --type fast
python run_tests.py --type unit
python run_tests.py --type slow
python run_tests.py --type all
```

