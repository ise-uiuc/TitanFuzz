# Coverage collection and program executor

1. Use multiprocessing to validate programs faster, and handles crashes gracefully (restart automatically).

2. Supports python coverage collection:
 - use `sys.settrace`

Run unit test:
```
python -m mycoverage.fuzztest
```
